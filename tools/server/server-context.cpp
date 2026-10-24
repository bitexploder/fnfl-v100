#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-splice.h"
#include "server-stream.h"
#include "server-tu.h"

#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "log.h"

// the engine: FNFL drives the model; this file keeps the server's tasks, slots, prefix matching, stop strings, streaming
// and metrics, and hands each slot's prefill and steps to it
#include "fn/fnfl.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cinttypes>
#include <exception>
#include <memory>
#include <filesystem>
#include <random>
#include <utility>
#include <fstream>
#include <mutex>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_STARTED,           // after assigning a task and before its prompt is read
    SLOT_STATE_PROCESSING_PROMPT, // MU1: its read is a job (prepared, its chunks running in turns)
    SLOT_STATE_GENERATING,
};

struct server_slot {
    int id;

    const llama_vocab * vocab = nullptr;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx = 0; // context size per slot

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;
    int32_t n_step_ahead  = 0; // SV1: tokens of the current engine step after the one being processed

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    // the tokens the engine holds for this slot, and the last generated token, which the next step feeds
    server_prompt prompt;
    llama_token   sampled = LLAMA_TOKEN_NULL;

    // BG2: the answers the slot generated to end-of-generation and still holds, ascending, for later turns' splices
    std::vector<server_answer> answers;

    // (ticket TU2, Amendment 1; MU1) the cached prefix a not-begun task's read will reuse, computed once per task
    // (tu_remaining), and when its last turn began (tu_order's rotation)
    int     tu_prefix_id_task = -1;
    int32_t tu_prefix         = 0;
    int64_t tu_last_read_seq  = 0;
    // MU1: the read job's prefix kept by the shell, its new tokens and the engine's processed count (start_prompt's),
    // its start time, and a cancellation the client asked for while it reads (BG1: honored at the chunk's end)
    int32_t tu_n_past      = 0;
    int32_t tu_n_new       = 0;
    int32_t tu_n_processed = 0;
    int64_t tu_t_start     = 0;
    bool    tu_cancel      = false;

    server_slot_stats stats;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;
        n_step_ahead   = 0;

        generated_tokens.clear();
        generated_token_probs.clear();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};

        n_predict_max = -1;
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(vocab, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }
};

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

    // (ticket TU2; MU1) the two-user scheduling policy's effective values, for GET /props (any thread)
    json tu_props_json() const {
        std::lock_guard<std::mutex> lock(tu_props_mutex);
        return tu_props;
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // the tokenizer (CPU only) and the engine
    llama_vocab *   vocab_own = nullptr;
    fnfl_engine *   engine    = nullptr;
    fnfl_model_meta model_meta;

    int32_t n_ctx_slot_val = 0;

    // slots / clients
    std::vector<server_slot> slots;

    // (tickets TU1..TU6; MU1) two-user scheduling, engaged whenever the server has more than one slot: a slot's read is
    // a job of layer groups (fnfl_read_*), the generating slots' steps run between its slices. The policy is runtime
    // state: slice_ms and group, with POST /props (TU2); TU6's fix and fair_read are permanent behavior
    struct tu_policy_t {
        int32_t slice_ms   = 100;  // a slice's budget
        // layers a group (the engine's, for chunks that start after a change). MU1's G2 preset: 2, not TU6's 4: at
        // 8,192 rows a group of 4 takes about 190 ms, twice the slice; 2 fits it (engineer's preset decision, MU1)
        int32_t group      = 2;
        // V1's fixed preset: the short-read class (uncached <= fast_limit first), TU6's reviewed n_batch, whatever
        // the launch's -b. A turn is one FNFL chunk (PF1's partition, the read's numerics), and a reader served out of
        // turn runs its chunk in the other-reader lane in slices, so TU6's n_ubatch cap has no counterpart
        int32_t fast_limit = 4096;
    } tu_policy;

    bool tu_engaged = false;
    // FMU1: resident TP (fnfl_resident_tp): the path is chosen from what is active each iteration (a lone writer the
    // one-slot body, a lone reader whole chunks, writers beside a reader 1-layer groups; admissions charged to slices;
    // the next reader's chunk staged ahead). Off (the one-card and cached configurations): MU1's and MXMU1's loop
    bool tu_resident = false;
    int32_t tu_version = 1; // +1 per accepted POST /props

    mutable std::mutex tu_props_mutex;
    json               tu_props = json::object(); // the effective values (tu_publish)

    // (ticket TU2) the reader to serve next, at the start of every iteration (tu_order)
    server_slot * tu_pick      = nullptr;
    server_slot * tu_pick_prev = nullptr;
    int64_t       tu_read_seq  = 0;
    std::vector<tu_reader> tu_readers; // FMU1: the iteration's readers (tu_order), for the next reader (tu_follow)

    // the jobs: [0] the live reader's chunk in the main arena; [1] (TU2's tu_read_other) a reader served out of turn,
    // its chunk in the other-reader lane while job 0's stays suspended. A job is one turn: its chunk, start to end
    struct tu_job_t {
        bool          active = false;
        server_slot * slot   = nullptr;

        int32_t n_rounds_pending = 0; // decode rounds since the job's last step
        int32_t peer_rounds      = 0; // decode rounds between the job's consecutive steps
        bool    overshoot_logged = false;

        double  service_ms       = 0.0; // the job's own wall time (its steps)
        double  longest_slice_ms = 0.0;
        double  last_group_ms    = -1.0; // the chunk's last measured group (-1 at its start)
        int32_t steps            = 0;
        int64_t groups_at_start  = 0;
        int32_t chunk            = 0, rows = 0, lane = -1;
    } tu_job[2];

    int slots_debug = 0; // env: LLAMA_SERVER_SLOTS_DEBUG

    server_metrics metrics;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    void destroy() {
        fnfl_release(engine);
        engine = nullptr;

        llama_vocab_free(vocab_own);
        vocab_own = nullptr;
        vocab     = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    void report_load_progress(float progress) {
        if (callback_state) {
            callback_state(SERVER_STATE_LOADING, {
                {"stages", std::vector<std::string>{"text_model"}},
                {"current", "text_model"},
                {"value", progress},
            });
        }
    }

    // the launch options FNFL does not serve stop the load with a clear message
    bool check_unsupported(const common_params & params) const {
        std::string what;
        if (params.embedding) {
            what = "embeddings and reranking (--embedding, --rerank)";
        } else if (!params.mmproj.path.empty() || !params.mmproj.url.empty()) {
            what = "multimodal input (--mmproj)";
        } else if (!params.lora_adapters.empty()) {
            what = "LoRA adapters (--lora)";
        }
        if (!what.empty()) {
            SRV_ERR("%s are not supported by FNFL\n", what.c_str());
            return false;
        }
        return true;
    }

    // load the tokenizer and the engine
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        const bool is_resume = sleeping;

        params_base = params;

        if (!check_unsupported(params_base)) {
            return false;
        }

        report_load_progress(0.0f);

        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        vocab_own = llama_vocab_load_from_file(params_base.model.path.c_str());
        if (vocab_own == nullptr) {
            SRV_ERR("failed to load the tokenizer of '%s'\n", params_base.model.path.c_str());
            return false;
        }
        vocab = vocab_own;

        // the end-of-generation biases a request's ignore_eos applies
        params_base.sampling.logit_bias_eog.clear();
        for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
            if (llama_vocab_is_eog(vocab, i)) {
                params_base.sampling.logit_bias_eog.push_back({i, -INFINITY});
            }
        }
        if (params_base.sampling.ignore_eos) {
            params_base.sampling.logit_bias.insert(params_base.sampling.logit_bias.end(),
                    params_base.sampling.logit_bias_eog.begin(), params_base.sampling.logit_bias_eog.end());
        }

        fnfl_load_params lp;
        lp.model_path = params_base.model.path;
        lp.draft_path = params_base.speculative.draft.mparams.path;
        lp.n_ctx      = params_base.n_ctx;
        lp.n_parallel = params_base.n_parallel;
        lp.n_batch    = params_base.n_batch;
        lp.n_ubatch   = params_base.n_ubatch;
        // TK1: the engine launches with the server's sampling arguments
        lp.temperature = params_base.sampling.temp;
        lp.top_k       = params_base.sampling.top_k;
        lp.top_p       = params_base.sampling.top_p;
        lp.min_p       = params_base.sampling.min_p;
        // SV1: every end-of-generation token of the vocabulary ends generation in the engine
        std::string eog_list;
        for (const auto & b : params_base.sampling.logit_bias_eog) {
            lp.eog_ids.push_back(b.token);
            eog_list += (eog_list.empty() ? "" : ",") + std::to_string(b.token);
        }
        lp.config.draft_subset   = params_base.fnfl.draft_subset;
        lp.config.ple_table      = params_base.fnfl.ple_table;
        lp.config.spec_d         = params_base.fnfl.spec_d;
        lp.config.draft_attention = params_base.fnfl.draft_attention;
        lp.config.check_state    = params_base.fnfl.check_state;
        lp.config.test_eog       = params_base.fnfl.test_eog;
        lp.config.test_fail_step = params_base.fnfl.test_fail_step;
        lp.config.single_card     = params_base.fnfl.single_card;
        lp.config.expert_pool_mib = params_base.fnfl.expert_pool_mib;
        lp.config.expert_cpu      = params_base.fnfl.expert_cpu;
        lp.config.cached_tp       = params_base.fnfl.cached_tp;
        lp.config.kv_host         = params_base.fnfl.kv_host;
        lp.config.device_cap_mib  = params_base.fnfl.device_cap_mib;
        lp.config.expert_profile      = params_base.fnfl.expert_profile;
        lp.config.expert_profile_save = params_base.fnfl.expert_profile_save;
        lp.config.expert_swap_overlap = params_base.fnfl.expert_swap_overlap;
        lp.config.settings            = params_base.fnfl.settings; // ME3
        lp.config.expert_reserve_mib  = params_base.fnfl.expert_reserve_mib;
        lp.config.kv_format           = params_base.fnfl.kv_format; // KV8
        SRV_INF("FNFL end-of-generation IDs: %s\n", eog_list.c_str());

        std::string err;
        engine = fnfl_load(lp, model_meta, err);
        if (engine == nullptr) {
            SRV_ERR("FNFL failed to load '%s': %s\n", params_base.model.path.c_str(), err.c_str());
            return false;
        }

        report_load_progress(1.0f);

        n_ctx_slot_val = model_meta.n_ctx_slot;
        if (params_base.kv_unified_per_slot > 0) {
            n_ctx_slot_val = std::min(n_ctx_slot_val, params_base.kv_unified_per_slot);
        }

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d\n", params_base.n_parallel, n_ctx_slot());

        slots.clear();
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id    = i;
            slot.vocab = vocab;
            slot.n_ctx = n_ctx_slot();

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                // MU1: the engine's slot is idle (its pages reclaimable); one slot keeps today's calls
                if (params_base.n_parallel > 1) {
                    fnfl_release_slot(engine, id_slot);
                }
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.reset();
        }

        // (MU1) two-user scheduling: engaged with more than one slot; the engine's group is the policy's
        tu_engaged = params_base.n_parallel > 1;
        if (tu_engaged && !fnfl_set_read_group(engine, tu_policy.group)) {
            SRV_ERR("FNFL refused the read group of %d layers\n", tu_policy.group);
            return false;
        }
        tu_resident = tu_engaged && fnfl_resident_tp(engine);
        tu_publish();
        if (tu_engaged) {
            SRV_INF("multi-user scheduling (FMU1): %s\n",
                    tu_resident ? "resident TP, the path chosen from the active slots (lone writer, lone reader, "
                                  "writers beside a reader, readers)"
                                : "MU1's turn-taking (not resident TP)");
            SRV_INF("two-user scheduling (MU1, TU6's port): engaged, %d slots, slice %d ms, group %d layers, short "
                    "reads up to %d uncached tokens (TU6's n_batch), a turn one chunk (main lane %d rows), "
                    "other-reader lane %d rows; KV pool %lld cells in pages of %d, each slot's context %d\n",
                    params_base.n_parallel, tu_policy.slice_ms, tu_policy.group, tu_policy.fast_limit,
                    fnfl_lane_rows(engine, 0), fnfl_lane_rows(engine, 1), (long long) model_meta.pool_cells,
                    model_meta.pool_page_cells, n_ctx_slot());
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(engine != nullptr);
        GGML_ASSERT(vocab  != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(vocab,
                        model_meta.chat_template.empty()          ? nullptr : model_meta.chat_template.c_str(),
                        model_meta.chat_template_tool_use.empty() ? nullptr : model_meta.chat_template_tool_use.c_str(),
                        params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store engine or tokenizer pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool specified = params_base.preserve_reasoning_specified;
                // note: the kwarg is enabled by default if not specified explicitly, so check the value
                bool enabled = it != params_base.default_template_kwargs.end() && it->second == "true";
                if (supported) {
                    SRV_TRC("preserve_reasoning kwarg: %s\n",
                            it == params_base.default_template_kwargs.end() ? "unset (template default)" : it->second.c_str());
                } else {
                    SRV_TRC("%s", "preserve_reasoning kwarg: not supported by template\n");
                }
                if (supported && !specified) {
                    SRV_WRN("%s", "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n");
                }
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && specified && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        // if a specific slot is requested, use it
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
                return ret;
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                // (ticket TU6) the common prefix over the slot's tokens
                const float f_keep = ret->prompt.tokens.empty()
                    ? 1.0f
                    : float(ret->prompt.tokens.get_common_prefix(task.tokens)) / float(ret->prompt.tokens.size());
                (void) f_sim_best;

                SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        f_sim_best, slot_prompt_similarity, f_keep);
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match ((ticket TU6) idle ties go to the lowest slot)
                if (!ret || slot.t_last_used < t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);
            }
        }

        return ret;
    }

    // SV1: the sampling options FNFL does not serve, each named when set to a non-neutral value; empty when none
    std::string fnfl_refused(const task_params & p) const {
        const common_params_sampling & s = p.sampling;
        std::vector<std::string> v;
        if (s.temp > 0.0f && (s.top_k <= 0 || s.top_k > 256)) {
            v.push_back("top_k " + std::to_string(s.top_k) + " (at temperature > 0 FNFL serves top-k 1 .. 256: the "
                        "full-vocabulary pick is not built)");
        }
        if (s.penalty_repeat != 1.0f) {
            v.push_back("repeat_penalty");
        }
        if (s.penalty_freq != 0.0f) {
            v.push_back("frequency_penalty");
        }
        if (s.penalty_present != 0.0f) {
            v.push_back("presence_penalty");
        }
        if (s.ignore_eos) {
            v.push_back("ignore_eos");
        }
        const size_t n_eog_bias = s.ignore_eos ? params_base.sampling.logit_bias_eog.size() : 0;
        if (s.logit_bias.size() > n_eog_bias) {
            v.push_back("logit_bias");
        }
        if (s.n_probs > 0) {
            v.push_back("n_probs (logprobs)");
        }
        if (s.dry_multiplier != 0.0f) {
            v.push_back("dry_multiplier (DRY)");
        }
        if (s.xtc_probability > 0.0f) {
            v.push_back("xtc_probability (XTC)");
        }
        if (s.typ_p != 1.0f) {
            v.push_back("typical_p");
        }
        if (s.top_n_sigma >= 0.0f) {
            v.push_back("top_n_sigma");
        }
        if (s.mirostat != 0) {
            v.push_back("mirostat");
        }
        if (s.dynatemp_range != 0.0f) {
            v.push_back("dynatemp_range (dynamic temperature)");
        }
        if (s.adaptive_target >= 0.0f) {
            v.push_back("adaptive_target");
        }
        if (s.min_keep != 0) {
            v.push_back("min_keep");
        }
        if (s.samplers != params_base.sampling.samplers) {
            v.push_back("samplers (a non-default sampler order)");
        }
        if (s.reasoning_budget_tokens >= 0) {
            v.push_back("reasoning_budget_tokens (thinking_budget_tokens)");
        }
        if (p.n_cmpl > 1) {
            v.push_back("n (" + std::to_string(p.n_cmpl) + " completions)");
        }
        std::string msg;
        for (const auto & f : v) {
            msg += (msg.empty() ? "" : ", ") + f;
        }
        return msg.empty() ? msg : "not supported by FNFL: " + msg;
    }

    // SV1 (--fnfl-check-state): the engine's tokens against the slot's after an engine call; false on a mismatch
    bool fnfl_check_state(server_slot & slot, const char * call) {
        if (!params_base.fnfl.check_state) {
            return true;
        }
        std::vector<int32_t> toks;
        int32_t pending = -1;
        const fnfl_status st = fnfl_slot_tokens(engine, slot.id, toks, pending);
        const llama_tokens & mine = slot.prompt.tokens.get_tokens();
        size_t i = 0;
        while (i < toks.size() && i < mine.size() && toks[i] == mine[i]) {
            ++i;
        }
        const bool same_pending = slot.state != SLOT_STATE_GENERATING || slot.sampled == LLAMA_TOKEN_NULL ||
                                  pending == slot.sampled;
        if (st == FNFL_OK && i == toks.size() && i == mine.size() && same_pending) {
            SLT_INF(slot, "FNFL state check: OK n=%zu (after %s)\n", toks.size(), call);
            return true;
        }
        SLT_ERR(slot, "FNFL state check: MISMATCH after %s: engine %zu tokens, slot %zu, first difference at %zu; pending "
                "engine %d slot %d\n", call, toks.size(), mine.size(), i, pending, slot.sampled);
        return false;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        if (!task.tokens.validate(vocab)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        const std::string refused = fnfl_refused(task.params);
        if (!refused.empty()) {
            send_error(task, refused, ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        splice_answers(slot, task);

        slot.task  = std::make_unique<const server_task>(std::move(task));
        slot.state = SLOT_STATE_STARTED;
        // (ticket TU2, Amendment 1) the prefix its read will reuse, once per task; no turn yet
        slot.tu_prefix_id_task = slot.task->id;
        slot.tu_prefix         = slot.task->params.cache_prompt ? (int32_t) slot.prompt.tokens.get_common_prefix(slot.task->tokens) : 0;
        slot.tu_last_read_seq  = 0;
        slot.tu_cancel         = false;

        SLT_INF(slot, "%s", "processing task\n");
        return true;
    }

    // BG1: a chat follow-up re-renders the slot's answers through the chat template, which need not give back the
    // tokens the model generated: the parser drops the whitespace around </think> and the tool calls and the template
    // re-inserts it as fixed strings, and the model may split a text into tokens other than the tokenizer's way. The
    // engine cannot rewind its state inside a turn, so such a follow-up re-prefilled from the prompt's checkpoint. When
    // the new prompt carries an answer unchanged in substance (its text, or failing that the message it parses to, is
    // the generated one's), the generated tokens replace the re-rendered ones. BG2: the client re-sends every earlier
    // answer in every request, so each answer the slot still holds is spliced, not only the last.
    void splice_answers(const server_slot & slot, server_task & task) {
        if (!task.params.cache_prompt || slot.prompt.tokens.empty()) {
            return;
        }
        const llama_tokens & held = slot.prompt.tokens.get_tokens();
        const llama_tokens & next = task.tokens.get_tokens();
        const size_t lcp = slot.prompt.tokens.get_common_prefix(task.tokens);
        if (lcp >= held.size() || lcp >= next.size()) {
            return; // an extension (or a prefix) of the slot
        }
        const auto piece = [this](const llama_tokens & t, size_t i) {
            return i < t.size() ? std::to_string(t[i]) + " '" + common_token_to_piece(vocab, t[i], true) + "'"
                                : std::string("(end)");
        };
        SLT_INF(slot, "BG1: the prompt leaves the slot at %zu of %zu: slot %s %s | prompt %s %s\n", lcp, held.size(),
                piece(held, lcp).c_str(), piece(held, lcp + 1).c_str(), piece(next, lcp).c_str(),
                piece(next, lcp + 1).c_str());

        const common_chat_parser_params & parser = task.params.chat_parser_params;
        std::string                       parse_error;
        const server_answer_same          same = [this, &parser, &parse_error](const llama_tokens & gen,
                                                                               const llama_tokens & ren) {
            const std::string gen_text = common_detokenize(vocab, gen, true);
            const std::string ren_text = common_detokenize(vocab, ren, true);
            if (gen_text == ren_text) {
                return true;
            }
            try {
                const common_chat_msg a = common_chat_parse(gen_text, false, parser);
                const common_chat_msg b = common_chat_parse(ren_text, false, parser);
                return !a.empty() && a.to_json_oaicompat().dump() == b.to_json_oaicompat().dump();
            } catch (const std::exception & e) {
                parse_error = e.what();
                return false;
            }
        };
        std::vector<server_splice_step> steps;
        const llama_tokens              out = server_splice_answers(held, next, slot.answers, same, steps);
        for (const server_splice_step & st : steps) {
            const server_answer & a    = slot.answers[st.index];
            std::string           what = "the prompt's copy differs from it; not spliced";
            if (st.spliced) {
                what = "its " + std::to_string(st.n_gen) + " generated tokens replace its " + std::to_string(st.n_ren) +
                       " re-rendered ones";
            } else if (st.n_ren == 0) {
                what = "no end-of-generation after it in the prompt; not spliced";
            } else if (!parse_error.empty()) {
                what = "the answer did not parse (" + parse_error + "); not spliced";
            }
            SLT_INF(slot, "BG2: answer %zu of %zu (slot %zu-%zu): %s\n", st.index + 1, slot.answers.size(), a.start,
                    a.end, what.c_str());
        }
        const size_t lcp_out = server_common_prefix(held, out, lcp);
        SLT_INF(slot, "BG2: %zu of %zu recorded answers reached; the prompt (%zu -> %zu tokens) %s at %zu of %zu\n",
                steps.size(), slot.answers.size(), next.size(), out.size(),
                lcp_out >= held.size() ? "extends the slot" : "leaves the slot", lcp_out, held.size());
        if (out != next) {
            task.tokens = server_tokens(out);
        }
    }

    // BG2: the slot's answer ended at end-of-generation: record it for the following turns' splices
    void record_answer(server_slot & slot) {
        const size_t start = (size_t) slot.task->n_tokens();
        const size_t end   = slot.prompt.tokens.size();
        if (slot.sampled == LLAMA_TOKEN_NULL || !llama_vocab_is_eog(vocab, slot.sampled) || start >= end) {
            return;
        }
        server_answers_keep_first(slot.answers, start);
        slot.answers.push_back({ start, end, slot.sampled });
    }

    // the slot keeps its first n tokens, and the answers it still holds whole
    void prompt_keep_first(server_slot & slot, size_t n) {
        slot.prompt.tokens.keep_first(n);
        server_answers_keep_first(slot.answers, n);
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        const std::string token_str = result.text_to_send;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // FNFL has no context shift: stop before the slot's context is full (SV1: at this token's position)
        if (slot.prompt.n_tokens() - slot.n_step_ahead + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // BG1: a prompt-progress event during the prefill: cache tokens kept, processed tokens held so far (cache included)
    void send_progress(server_slot & slot, int32_t cache, int32_t processed) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id                    = slot.task->id;
        res->index                 = slot.task->index;
        res->is_progress           = true;
        res->progress.total        = slot.task->n_tokens();
        res->progress.cache        = cache;
        res->progress.processed    = processed;
        res->progress.time_ms      = slot.stats.t_elapsed_us() / 1000;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = cache;
        res->verbose               = slot.task->params.verbose;
        res->res_type              = slot.task->params.res_type;
        res->oaicompat_model       = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id     = slot.task->params.oaicompat_cmpl_id;

        queue_results.send(std::move(res));
    }

    struct prefill_watch {
        server_context_impl * ctx;
        server_slot *         slot;
    };

    // BG1: fnfl_prefill's progress callback, on this thread at each chunk's end: report, and stop if the client left
    static bool on_prefill_chunk(void * user, int32_t from, int32_t done, int32_t total) {
        GGML_UNUSED(total);
        auto * w = static_cast<prefill_watch *>(user);
        server_slot & slot = *w->slot;
        if (slot.task->params.stream && slot.task->params.return_progress) {
            w->ctx->send_progress(slot, from, done);
        }
        return !w->ctx->queue_tasks.has_cancel(slot.task->id);
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(vocab, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(vocab, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // returns false to decline the task, it is offered again after the current step is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("busy, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                {
                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            // (ticket TU1, BG1; MU1) a reading slot: its job finishes the chunk in progress, the other
                            // slots served between its slices, then stops; the slot is released then (tu_slice)
                            if (slot.state == SLOT_STATE_PROCESSING_PROMPT) {
                                fnfl_read_cancel(engine, slot.id);
                                slot.tu_cancel = true;
                                SLT_INF(slot, "%s", "cancel during the read: the job stops at its chunk's end (BG1)\n");
                                break;
                            }
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_TU_POLICY:
                {
                    // (ticket TU2) POST /props, validated by the HTTP thread; applied here, between steps (never while
                    // yielding): a living job takes the new slice at its next slice, a new group the chunks begun after
                    const auto & set = task.tu_set;
                    if (!tu_engaged) {
                        send_error(task, "two-user scheduling needs more than one slot (--parallel 2 .. 4)", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (set.slice_ms > 0) {
                        tu_policy.slice_ms = set.slice_ms;
                    }
                    if (set.group > 0) {
                        const int32_t g = std::min<int32_t>(set.group, 48); // clamped to the layer count
                        if (fnfl_set_read_group(engine, g)) {
                            tu_policy.group = g;
                        }
                    }
                    tu_version++;
                    LOG_WRN("TU policy: version %d, slice %d ms, group %d layers (POST /props)\n", tu_version,
                            tu_policy.slice_ms, tu_policy.group);
                    tu_publish();

                    auto res = std::make_unique<server_task_result_tu_policy>();
                    res->id = task.id;
                    res->tu = get_tu_props();
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    // KV8: the engine's own count of each slot's held cells, and the pool's mapped cells (the
                    // shell's prompt record syncs only when a task starts on the slot)
                    fnfl_pool_report pool_rep;
                    const bool have_pool = engine != nullptr && fnfl_pool(engine, pool_rep) && pool_rep.on;
                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        json js = slot.to_json(slots_debug == 0);
                        if (engine != nullptr) {
                            js["fnfl_held"] = fnfl_slot_held(engine, slot.id);
                            if (have_pool && slot.id >= 0 && slot.id < 4) {
                                js["fnfl_pool_cells"] = (int64_t) pool_rep.mapped[slot.id] * pool_rep.page_cells;
                            }
                        }
                        slots_data.push_back(std::move(js));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    prompt_clear(*slot);
                    slot->t_last_used = -1; // (ticket TU6) an erased slot is the least recently used

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    // the engine and the slot forget the slot's tokens
    void prompt_clear(server_slot & slot) {
        SLT_TRC(slot, "clearing prompt with %zu tokens\n", slot.prompt.tokens.size());

        fnfl_truncate(engine, slot.id, 0);
        slot.prompt.clear();
        slot.answers.clear();
    }

    // an engine call failed: the request gets the engine's reason, and the slot is released
    void engine_failed(server_slot & slot, fnfl_status status) { engine_failed(slot, status, fnfl_last_error(engine)); }

    // (FMU2) a slot's failure with its own reason (a joint step's)
    void engine_failed(server_slot & slot, fnfl_status status, const std::string & why) {
        send_error(slot, std::string(fnfl_status_str(status)) + (why.empty() ? "" : ": " + why),
                   status == FNFL_NOT_BUILT    ? ERROR_TYPE_NOT_SUPPORTED
                   : status == FNFL_NO_CONTEXT ? ERROR_TYPE_EXCEED_CONTEXT_SIZE
                                               : ERROR_TYPE_SERVER);
        prompt_clear(slot);
        slot.release();
    }

    // a started slot's prompt: reuse the cached prefix, then the engine reads the rest
    void start_prompt(server_slot & slot) {
        const auto & input_tokens = slot.task->tokens;

        slot.stats.update_prompt_start();

        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, task.n_tokens = %d\n", slot.n_ctx, slot.task->n_tokens());

        // empty prompt passed -> release the slot and send empty response
        if (input_tokens.empty()) {
            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

            slot.stats.update_prompt_last();
            slot.print_timings();
            send_final_response(slot);
            slot.release();

            return;
        }

        if (slot.task->n_tokens() >= slot.n_ctx) {
            send_error(slot,
                       string_format("request (%d tokens) exceeds the available context size (%d "
                                     "tokens), try increasing it",
                                     slot.task->n_tokens(), slot.n_ctx),
                       ERROR_TYPE_EXCEED_CONTEXT_SIZE);
            slot.release();
            return;
        }

        // SV1: the request's sampling and token budget, before its prefill; a seed of -1 draws one per request
        {
            const common_params_sampling & sp = slot.task->params.sampling;
            fnfl_sampling fs;
            fs.seed        = sp.seed == LLAMA_DEFAULT_SEED ? (uint64_t(std::random_device{}()) << 32) | std::random_device{}()
                                                           : uint64_t(sp.seed);
            fs.temperature = sp.temp;
            fs.top_k       = sp.top_k;
            fs.top_p       = sp.top_p;
            fs.min_p       = sp.min_p;
            fs.max_tokens  = slot.n_predict_max;
            fs.n_ctx       = slot.n_ctx;
            std::string err;
            if (fnfl_set_sampling(engine, slot.id, fs, err) != FNFL_OK) {
                send_error(slot, "FNFL: " + err, ERROR_TYPE_INVALID_REQUEST);
                slot.release();
                return;
            }
        }

        // reuse the tokens the slot already holds that are common with the new prompt
        int32_t n_past = slot.task->params.cache_prompt ? (int32_t) slot.prompt.tokens.get_common_prefix(input_tokens) : 0;

        // at least one token is read, so the step has its row
        if (n_past == slot.task->n_tokens()) {
            n_past--;
        }

        const fnfl_status st_trunc = fnfl_truncate(engine, slot.id, n_past);
        if (st_trunc != FNFL_OK) {
            engine_failed(slot, st_trunc);
            return;
        }
        prompt_keep_first(slot, n_past);
        if (!fnfl_check_state(slot, "truncate")) {
            send_error(slot, "FNFL state check: mismatch after truncate", ERROR_TYPE_SERVER);
            prompt_clear(slot);
            slot.release();
            return;
        }

        slot.stats.n_prompt_cached    = n_past;
        slot.stats.n_prompt_processed = 0;

        const llama_tokens & all = input_tokens.get_tokens();
        int32_t n_new = slot.task->n_tokens() - n_past;

        // BG1: the client hears at once that its prompt is being read (the HTTP status goes out with this first event,
        // and the SSE pings follow), then the prefill's progress at each chunk's end
        if (slot.task->params.stream) {
            if (slot.task->params.return_progress) {
                send_progress(slot, n_past, n_past);
            } else {
                // for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                send_partial_response(slot, {}, false, true);
            }
        }

        const int64_t t_start = ggml_time_us();
        int32_t n_processed = n_new;
        fnfl_status st = FNFL_OK;
        prefill_watch watch = { this, &slot };
        // BG1: tasks that only read the server's state (/slots, /metrics) are answered during the prefill; the rest wait
        queue_tasks.yield_to_queue([&]() {
            st = fnfl_prefill(engine, slot.id, all.data() + n_past, n_new, n_past, &n_processed, &on_prefill_chunk, &watch);
        });
        if (st == FNFL_CANCELLED) {
            // the client left during the prefill: the slot keeps the prompt's tokens the engine kept (n_processed)
            const int32_t kept = std::min(n_processed, slot.task->n_tokens());
            if (kept <= n_past) {
                prompt_keep_first(slot, kept);
            } else {
                slot.prompt.tokens.insert(llama_tokens(all.begin() + n_past, all.begin() + kept));
            }
            SLT_INF(slot, "prefill cancelled by the client; the slot keeps %d of the prompt's %d tokens\n", kept,
                    slot.task->n_tokens());
            slot.sampled = LLAMA_TOKEN_NULL;
            slot.release();
            return;
        }
        if (st != FNFL_OK) {
            engine_failed(slot, st);
            return;
        }
        // SV1, PF1: the engine re-prefills from its latest checkpoint at or before the kept prefix (token 0 when none
        // fits) when that prefix is not its processed length; count what it did
        if (n_processed != n_new) {
            slot.stats.n_prompt_cached = slot.task->n_tokens() - n_processed;
            n_new = n_processed;
        }
        metrics.add_prompt_cached(slot.stats.n_prompt_cached);

        slot.prompt.tokens.insert(llama_tokens(all.begin() + n_past, all.end()));
        if (!fnfl_check_state(slot, "prefill")) {
            send_error(slot, "FNFL state check: mismatch after prefill", ERROR_TYPE_SERVER);
            prompt_clear(slot);
            slot.release();
            return;
        }

        slot.stats.n_prompt_processed = n_new;
        slot.stats.update_prompt_last();
        metrics.add_prompt(n_new, ggml_time_us() - t_start);

        // the prompt's last progress event: all of it processed
        if (slot.task->params.stream && slot.task->params.return_progress) {
            send_partial_response(slot, {}, true);
        }

        slot.stats.n_gen = 0;
        slot.sampled     = LLAMA_TOKEN_NULL;
        slot.state       = SLOT_STATE_GENERATING;
    }

    // one engine step of a generating slot: 1-5 accepted tokens (SP2)
    void step_slot(server_slot & slot) {
        fnfl_step_result r;
        const fnfl_status st = fnfl_step(engine, slot.id, r);
        if (st != FNFL_OK) {
            engine_failed(slot, st);
            return;
        }
        apply_step(slot, r);
    }

    // a step's tokens into the slot: the state check, each token's processing and the request's end
    void apply_step(server_slot & slot, const fnfl_step_result & r) {
        if (r.n_tokens < 1) {
            engine_failed(slot, FNFL_ERROR);
            return;
        }

        // the engine state now holds every token but the last of this step, which the next step feeds
        if (slot.sampled != LLAMA_TOKEN_NULL) {
            slot.prompt.tokens.push_back(slot.sampled);
        }
        slot.prompt.tokens.insert(llama_tokens(r.tokens, r.tokens + r.n_tokens - 1));
        slot.sampled = r.tokens[r.n_tokens - 1];
        if (!fnfl_check_state(slot, "step")) {
            send_error(slot, "FNFL state check: mismatch after step", ERROR_TYPE_SERVER);
            prompt_clear(slot);
            slot.release();
            return;
        }

        const bool accept_special = params_base.special;

        for (int32_t i = 0; i < r.n_tokens; ++i) {
            const llama_token tok = r.tokens[i];
            slot.n_step_ahead = r.n_tokens - 1 - i; // the step's tokens after this one are already in the slot

            slot.stats.n_gen += 1;
            if (slot.stats.n_gen == 1) {
                slot.t_print_last = ggml_time_us();
                slot.n_gen_last   = 0;
            }
            slot.stats.update_gen_last();

            completion_token_output result;
            result.tok          = tok;
            result.text_to_send = common_token_to_piece(vocab, tok,
                    accept_special || slot.task->params.sampling.preserved_tokens.count(tok) > 0);
            result.prob         = 1.0f;

            if (!process_token(result, slot) || (r.end && i + 1 == r.n_tokens)) {
                if (slot.stop == STOP_TYPE_NONE) {
                    slot.stop = STOP_TYPE_EOS;
                }
                record_answer(slot);
                slot.print_timings();
                send_final_response(slot);
                slot.release();
                return;
            }
        }

        slot.print_timings_tg();
    }

    // ---- (tickets TU1..TU6; MU1) two-user scheduling ----

    json get_tu_props() const {
        std::lock_guard<std::mutex> lock(tu_props_mutex);
        return tu_props;
    }

    // (ticket TU2) the effective values, for GET /props and the POST /props response
    void tu_publish() {
        fnfl_pool_report pr;
        const bool pool = engine != nullptr && fnfl_pool(engine, pr);
        std::lock_guard<std::mutex> lock(tu_props_mutex);
        tu_props = {
            {"enabled", tu_engaged},
            {"presence", tu_resident},
            {"slice_ms", tu_policy.slice_ms},
            {"group", tu_policy.group},
            {"fast_limit", tu_policy.fast_limit},
            {"version", tu_version},
            {"slots", params_base.n_parallel},
            {"n_ctx_slot", n_ctx_slot()},
            {"kv_pool",
             json{
                 {"on", pool && pr.on},
                 {"cells", pool ? pr.capacity_cells : 0},
                 {"page_cells", pool ? pr.page_cells : 0},
                 {"pages", pool ? pr.pages : 0},
             }},
        };
    }

    // a slot other than the job's generates (the peer whose steps run between the job's slices)
    bool tu_peer(const server_slot * job_slot) const {
        for (const auto & slot : slots) {
            if (&slot != job_slot && slot.state == SLOT_STATE_GENERATING) {
                return true;
            }
        }
        return false;
    }

    // (ticket TU2) the slot's task can be a job (TU1's rule; FNFL serves completions of one text prompt only)
    bool tu_eligible(const server_slot & slot) const {
        return slot.task && (slot.task->type == SERVER_TASK_TYPE_COMPLETION || slot.task->type == SERVER_TASK_TYPE_INFILL);
    }

    // (ticket TU2, Amendment 1) prompt tokens the reader has left: a slot not begun counts its prompt less the prefix its
    // begin will reuse (once per task); a begun read its prompt less the tokens the engine has committed
    int32_t tu_remaining(server_slot & slot) const {
        if (slot.state == SLOT_STATE_STARTED && slot.tu_prefix_id_task != slot.task->id) {
            slot.tu_prefix_id_task = slot.task->id;
            slot.tu_prefix = slot.task->params.cache_prompt ? (int32_t) slot.prompt.tokens.get_common_prefix(slot.task->tokens) : 0;
        }
        int32_t committed = 0;
        if (slot.state == SLOT_STATE_PROCESSING_PROMPT) {
            fnfl_read_report r;
            if (fnfl_read_info(engine, slot.id, r) && r.live) {
                committed = r.committed;
            }
        }
        return tu_remaining_tokens(slot.state == SLOT_STATE_PROCESSING_PROMPT, slot.task->n_tokens(), slot.tu_prefix, committed);
    }

    // (ticket TU2, TU6) the eligible readers sorted by priority, short first, then the live reader and turn order
    // (long) or remaining (short); the first is the reader to serve next. One line when the pick or the readers change
    void tu_order() {
        tu_pick = nullptr;
        const int32_t fast_limit = tu_policy.fast_limit;
        std::vector<tu_reader> readers;
        for (auto & slot : slots) {
            const bool reading = slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT;
            if (reading && tu_eligible(slot)) {
                tu_reader r;
                r.slot      = slot.id;
                r.priority  = slot.task->params.priority;
                r.remaining = tu_remaining(slot);
                r.uncached  = tu_uncached(r.remaining, slot.task->n_tokens(), slot.tu_prefix);
                r.last_seq  = slot.tu_last_read_seq;
                r.id_task   = slot.task->id;
                r.live      = tu_job[0].active && tu_job[0].slot == &slot;
                r.begun = slot.state == SLOT_STATE_PROCESSING_PROMPT;
                readers.push_back(r);
            }
        }
        tu_readers = readers;
        if (readers.empty()) {
            tu_pick_prev = nullptr;
            return;
        }
        tu_sort(readers, fast_limit);
        tu_pick = &slots[size_t(readers[0].slot)];

        static int  last_pick_slot = -1;
        static int  last_n_readers = 0;
        static int  last_task_0    = -1;
        static int  last_task_1    = -1;
        static bool last_job_act   = false;
        const int cur_task_1 = readers.size() > 1 ? readers[1].id_task : -1;
        if (tu_pick->id != last_pick_slot || (int) readers.size() != last_n_readers || readers[0].id_task != last_task_0 ||
            cur_task_1 != last_task_1 || tu_job[0].active != last_job_act) {
            std::string r_str;
            for (size_t i = 0; i < readers.size(); ++i) {
                r_str += (i > 0 ? ";" : "") + std::string("s") + std::to_string(readers[i].slot) +
                         "(t=" + std::to_string(readers[i].id_task) + ",p=" + std::to_string(readers[i].priority) +
                         ",rem=" + std::to_string(readers[i].remaining) + ",unc=" + std::to_string(readers[i].uncached) +
                         ",seq=" + std::to_string(readers[i].last_seq) + ")";
            }
            LOG_WRN("TU order: t_us=%" PRId64 " job0=%d job1=%d pick=%d readers=[%s]\n", ggml_time_us(),
                    tu_job[0].active ? tu_job[0].slot->id : -1, tu_job[1].active ? tu_job[1].slot->id : -1,
                    tu_pick->id, r_str.c_str());
            last_pick_slot = tu_pick->id;
            last_n_readers = (int) readers.size();
            last_task_0    = readers[0].id_task;
            last_task_1    = cur_task_1;
            last_job_act   = tu_job[0].active;
        }
        if (tu_pick_prev != nullptr && tu_pick_prev != tu_pick) {
            for (const auto & r : readers) {
                if (r.slot == tu_pick_prev->id) {
                    LOG_WRN("TU pick: slot %d (priority %d, remaining %d) ahead of %sslot %d (remaining %d)\n",
                            tu_pick->id, readers[0].priority, readers[0].remaining,
                            tu_job[0].active && tu_job[0].slot == tu_pick_prev ? "the job's " : "", r.slot, r.remaining);
                    break;
                }
            }
        }
        tu_pick_prev = tu_pick;
    }

    // a started slot's read begun as a job (start_prompt's first half): the request's sampling, the reused prefix
    // (capped at what the engine still holds: a reclaimed slot holds nothing), the progress headers, then
    // fnfl_read_begin. false: the slot was answered and released
    bool tu_admit(server_slot & slot) {
        const auto & input_tokens = slot.task->tokens;
        slot.stats.update_prompt_start();
        if (input_tokens.empty()) {
            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");
            slot.stats.update_prompt_last();
            slot.print_timings();
            send_final_response(slot);
            slot.release();
            return false;
        }
        if (slot.task->n_tokens() >= slot.n_ctx) {
            send_error(slot,
                       string_format("request (%d tokens) exceeds the available context size (%d "
                                     "tokens), try increasing it",
                                     slot.task->n_tokens(), slot.n_ctx),
                       ERROR_TYPE_EXCEED_CONTEXT_SIZE);
            slot.release();
            return false;
        }
        {
            const common_params_sampling & sp = slot.task->params.sampling;
            fnfl_sampling fs;
            fs.seed        = sp.seed == LLAMA_DEFAULT_SEED ? (uint64_t(std::random_device{}()) << 32) | std::random_device{}()
                                                           : uint64_t(sp.seed);
            fs.temperature = sp.temp;
            fs.top_k       = sp.top_k;
            fs.top_p       = sp.top_p;
            fs.min_p       = sp.min_p;
            fs.max_tokens  = slot.n_predict_max;
            fs.n_ctx       = slot.n_ctx;
            std::string err;
            if (fnfl_set_sampling(engine, slot.id, fs, err) != FNFL_OK) {
                send_error(slot, "FNFL: " + err, ERROR_TYPE_INVALID_REQUEST);
                slot.release();
                return false;
            }
        }
        // MU1: the engine may hold fewer tokens than the slot's record (its pages were reclaimed while it was idle)
        const int32_t held = fnfl_slot_held(engine, slot.id);
        if (held < (int32_t) slot.prompt.tokens.size()) {
            SLT_INF(slot, "the KV pool reclaimed the slot's cached prompt: the engine holds %d of its %zu tokens\n", held,
                    slot.prompt.tokens.size());
            prompt_keep_first(slot, held);
            slot.answers.clear();
        }
        int32_t n_past = slot.task->params.cache_prompt ? (int32_t) slot.prompt.tokens.get_common_prefix(input_tokens) : 0;
        if (n_past == slot.task->n_tokens()) {
            n_past--;
        }
        const fnfl_status st_trunc = fnfl_truncate(engine, slot.id, n_past);
        if (st_trunc != FNFL_OK) {
            engine_failed(slot, st_trunc);
            return false;
        }
        prompt_keep_first(slot, n_past);
        if (!fnfl_check_state(slot, "truncate")) {
            send_error(slot, "FNFL state check: mismatch after truncate", ERROR_TYPE_SERVER);
            prompt_clear(slot);
            slot.release();
            return false;
        }
        slot.stats.n_prompt_cached    = n_past;
        slot.stats.n_prompt_processed = 0;
        const llama_tokens & all = input_tokens.get_tokens();
        const int32_t n_new = slot.task->n_tokens() - n_past;
        if (slot.task->params.stream) {
            if (slot.task->params.return_progress) {
                send_progress(slot, n_past, n_past);
            } else {
                send_partial_response(slot, {}, false, true);
            }
        }
        int32_t n_processed = n_new;
        const int32_t r = fnfl_read_begin(engine, slot.id, all.data() + n_past, n_new, n_past, &n_processed);
        if (r == 1) {
            send_error(slot, "FNFL: " + fnfl_last_error(engine) + " (the KV pool or the context cannot hold the request)",
                       ERROR_TYPE_EXCEED_CONTEXT_SIZE);
            slot.release();
            return false;
        }
        if (r != 0) {
            engine_failed(slot, FNFL_ERROR);
            return false;
        }
        slot.tu_n_past      = n_past;
        slot.tu_n_new       = n_new;
        slot.tu_n_processed = n_processed;
        slot.tu_t_start     = ggml_time_us();
        slot.state          = SLOT_STATE_PROCESSING_PROMPT;
        SLT_INF(slot, "read job begun: %d tokens kept, %d new, %d to process\n", n_past, n_new, n_processed);
        return true;
    }

    // the generating slots' steps, each once, in slot order (decode first: TU1's round before any slice)
    std::string tu_stepped; // the iteration's stepped slots, for its line (tu_update)
    void tu_decode() {
        int32_t n = 0;
        tu_stepped.clear();
        // (FMU2) two or more generating slots step together: one call, the engine's joint round for those it can take
        std::vector<server_slot *> gen;
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_GENERATING) {
                gen.push_back(&slot);
            }
        }
        if (gen.size() >= 2 && gen.size() <= 4) {
            int32_t ids[4] = {};
            fnfl_step_result res[4];
            fnfl_status st[4] = {};
            std::string why[4];
            for (size_t k = 0; k < gen.size(); ++k) {
                ids[k] = gen[k]->id;
                metrics_on_step();
            }
            const fnfl_status call = fnfl_step_joint(engine, ids, int32_t(gen.size()), res, st, why);
            for (size_t k = 0; k < gen.size(); ++k) {
                server_slot & slot = *gen[k];
                try {
                    if (call != FNFL_OK || st[k] != FNFL_OK) {
                        engine_failed(slot, call != FNFL_OK ? call : st[k], call != FNFL_OK ? "step_joint" : why[k]);
                    } else {
                        apply_step(slot, res[k]);
                    }
                    tu_stepped += (n > 0 ? "," : "") + std::to_string(slot.id);
                    ++n;
                } catch (const std::exception & e) {
                    SLT_ERR(slot, "got exception: %s\n", e.what());
                    send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                    slot.release();
                }
            }
            gen.clear();
        }
        for (server_slot * sp : gen) {
            server_slot & slot = *sp;
            try {
                metrics_on_step();
                step_slot(slot);
                tu_stepped += (n > 0 ? "," : "") + std::to_string(slot.id);
                ++n;
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
        for (auto & j : tu_job) {
            if (j.active) {
                j.n_rounds_pending += n;
            }
        }
    }

    // (ticket TU1) a job (one turn: the reader's next chunk) begins on slot k's lane and runs its first slice;
    // (ticket TU2) the turn's order counter is taken when its rows begin
    void tu_start(server_slot & slot, int k) {
        tu_job_t & j = tu_job[k];
        j = tu_job_t();
        j.active = true;
        j.slot   = &slot;
        fnfl_read_report r;
        fnfl_read_info(engine, slot.id, r);
        j.groups_at_start = r.groups_done;
        j.chunk           = r.chunk;
        j.rows            = r.next_rows;
        slot.tu_last_read_seq = ++tu_read_seq;
        tu_slice(k);
    }

    // one slice of job k: groups while the slice's time so far plus the next step's predicted time stays within it
    // (TU3's one-sync loop: a step's groups synchronize once at its end), at least one group
    void tu_slice(int k) {
        tu_job_t & j = tu_job[k];
        server_slot & slot = *j.slot;
        const bool peer = tu_peer(&slot);
        // FMU1: the presence rule (resident TP): writers beside a reader one step a slice of main's group in 1-layer
        // groups (FMU2's reader rule); MU1's shared slices otherwise. FMU2's correction: a lone reader runs main's
        // geometry, one step a call, so an arrival waits one step, not the chunk (FMU1's lone chunk was one step)
        const tu_mode mode = tu_resident ? tu_slot_mode(slot) : tu_mode::shared;
        const fnfl_read_mode rmode = mode == tu_mode::writers ? FNFL_READ_WRITERS : FNFL_READ_SHARED;
        const double budget = tu_mode_budget(mode, tu_policy.slice_ms);
        double  t_slice      = 0.0;
        int32_t n_run        = 0;
        int64_t n_groups_run = 0;
        fnfl_read_status ret = FNFL_READ_YIELDED;
        while (true) {
            // FMU2's reader rule: beside writers, one step a slice
            const int32_t n_groups = mode == tu_mode::writers
                                         ? (n_run > 0 ? 0 : tu_writers_groups(budget, j.last_group_ms, tu_policy.group))
                                         : tu_next_groups(budget, t_slice, j.last_group_ms, n_run);
            if (n_groups < 1) {
                break;
            }
            fnfl_read_report before;
            fnfl_read_info(engine, slot.id, before);
            if (before.groups_done > j.groups_at_start) {
                j.peer_rounds += j.n_rounds_pending;
            }
            j.n_rounds_pending = 0;
            const int64_t t0 = ggml_time_us();
            queue_tasks.yield_to_queue([&]() { ret = fnfl_read_step(engine, slot.id, n_groups, rmode); });
            const double dt = (ggml_time_us() - t0) / 1000.0;
            fnfl_read_report after;
            fnfl_read_info(engine, slot.id, after);
            const int64_t ran = std::max<int64_t>(0, after.groups_done - before.groups_done);
            if (j.lane < 0) {
                j.lane = before.in_chunk ? before.lane : after.lane;
            }
            t_slice += dt;
            n_run++;
            n_groups_run += ran;
            j.steps++;
            j.service_ms += dt;
            j.last_group_ms = ret == FNFL_READ_YIELDED && ran > 0 ? dt / ran : -1.0;
            if (ret != FNFL_READ_YIELDED || mode == tu_mode::lone) {
                break;
            }
        }
        j.longest_slice_ms = std::max(j.longest_slice_ms, t_slice);
        if (peer && n_run == 1 && n_groups_run == 1 && t_slice > budget && !j.overshoot_logged) {
            j.overshoot_logged = true;
            LOG_WRN("TU job: slot %d, a group of %.1f ms overshoots the %d ms slice\n", slot.id, t_slice, tu_policy.slice_ms);
        }
        switch (ret) {
            case FNFL_READ_YIELDED:
                break;
            case FNFL_READ_CHUNK_DONE:
                {
                    // BG1: the prompt's progress at the chunk's end (where the read started, what the slot holds)
                    fnfl_read_report r;
                    if (slot.task->params.stream && slot.task->params.return_progress && fnfl_read_info(engine, slot.id, r)) {
                        send_progress(slot, r.from, r.committed);
                    }
                    tu_turn_end(k, "chunk");
                } break;
            case FNFL_READ_DONE:
                tu_turn_end(k, "read");
                tu_finish(slot);
                break;
            case FNFL_READ_CANCELLED:
                {
                    tu_turn_end(k, "cancelled");
                    LOG_WRN("TU job: slot %d cancelled at its chunk's end\n", slot.id);
                    tu_cancelled(slot);
                } break;
            default:
                {
                    tu_turn_end(k, "failed");
                    SLT_ERR(slot, "%s", "two-user scheduling: a step of the read job failed\n");
                    engine_failed(slot, FNFL_ERROR);
                } break;
        }
    }

    // FMU1: the presence rule for slot's read step: the generating slots, and the other slots reading or admitted
    tu_mode tu_slot_mode(const server_slot & reader) const {
        int32_t n_writers = 0, n_readers = 0;
        for (const auto & slot : slots) {
            if (&slot == &reader) {
                continue;
            }
            n_writers += slot.state == SLOT_STATE_GENERATING ? 1 : 0;
            n_readers += slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT ? 1 : 0;
        }
        return tu_presence(n_writers, n_readers);
    }

    // FMU1 (P): the reader whose turn follows the main lane's current one (in progress, or the pick's about to start)
    // has its next chunk staged ahead while the turn and the writers' rounds run
    void tu_prefetch() {
        server_slot * now = tu_job[0].active                                                       ? tu_job[0].slot
                            : tu_pick != nullptr && tu_pick->state == SLOT_STATE_PROCESSING_PROMPT ? tu_pick
                                                                                                   : nullptr;
        const int next = tu_follow(tu_readers, now != nullptr ? now->id : -1, tu_policy.fast_limit);
        if (!fnfl_read_prefetch(engine, now != nullptr ? now->id : -1, next)) {
            SRV_WRN("FNFL read prefetch failed: %s\n", fnfl_last_error(engine).c_str());
        }
    }

    // the turn's line (the job's identity, chunk, groups, peer rounds, service and longest slice), and the job's end
    void tu_turn_end(int k, const char * how) {
        tu_job_t & j = tu_job[k];
        fnfl_read_report r;
        fnfl_read_info(engine, j.slot->id, r);
        LOG_WRN("TU job: slot %d, task %d, %s: chunk %d (%d rows), lane %d, groups %lld of %d layers, peer rounds %d, "
                "steps %d, service ms %.1f, longest slice ms %.1f (slice %d ms); read %d of %d tokens committed\n",
                j.slot->id, j.slot->task ? j.slot->task->id : -1, how, j.chunk, j.rows, j.lane,
                (long long) (r.groups_done - j.groups_at_start), r.layers, j.peer_rounds, j.steps, j.service_ms,
                j.longest_slice_ms, tu_policy.slice_ms, r.committed, r.total);
        j.active = false;
        j.slot   = nullptr;
    }

    // the read is done: start_prompt's second half (the slot's tokens, stats, the last progress event), then generating
    void tu_finish(server_slot & slot) {
        const llama_tokens & all = slot.task->tokens.get_tokens();
        int32_t n_new = slot.tu_n_new;
        if (slot.tu_n_processed != n_new) {
            slot.stats.n_prompt_cached = slot.task->n_tokens() - slot.tu_n_processed;
            n_new = slot.tu_n_processed;
        }
        metrics.add_prompt_cached(slot.stats.n_prompt_cached);
        slot.prompt.tokens.insert(llama_tokens(all.begin() + slot.tu_n_past, all.end()));
        if (!fnfl_check_state(slot, "prefill")) {
            send_error(slot, "FNFL state check: mismatch after prefill", ERROR_TYPE_SERVER);
            prompt_clear(slot);
            slot.release();
            return;
        }
        slot.stats.n_prompt_processed = n_new;
        slot.stats.update_prompt_last();
        metrics.add_prompt(n_new, ggml_time_us() - slot.tu_t_start);
        if (slot.task->params.stream && slot.task->params.return_progress) {
            send_partial_response(slot, {}, true);
        }
        slot.stats.n_gen = 0;
        slot.sampled     = LLAMA_TOKEN_NULL;
        slot.state       = SLOT_STATE_GENERATING;
    }

    // BG1: the client left during the read: the slot keeps the prompt's tokens the engine kept
    void tu_cancelled(server_slot & slot) {
        const llama_tokens & all = slot.task->tokens.get_tokens();
        const int32_t kept = std::min(fnfl_slot_held(engine, slot.id), slot.task->n_tokens());
        if (kept <= slot.tu_n_past) {
            prompt_keep_first(slot, kept);
        } else {
            slot.prompt.tokens.insert(llama_tokens(all.begin() + slot.tu_n_past, all.begin() + kept));
        }
        SLT_INF(slot, "prefill cancelled by the client; the slot keeps %d of the prompt's %d tokens\n", kept,
                slot.task->n_tokens());
        slot.sampled = LLAMA_TOKEN_NULL;
        slot.release();
    }

    // (ticket TU2) another reader is to be served next while job 0's chunk is suspended: its next chunk in the
    // other-reader lane when it fits (else it waits for job 0's turn to end: the lane holds chunks of at most
    // fnfl_lane_rows(1) rows, and FNFL's chunk partition is the read's numerics)
    bool tu_other_fits(server_slot & slot) {
        if (slot.state == SLOT_STATE_STARTED) {
            return false; // begun by tu_admit first
        }
        fnfl_read_report r;
        if (!fnfl_read_info(engine, slot.id, r) || !r.live || r.in_chunk) {
            return false;
        }
        const bool fits = r.next_rows <= fnfl_lane_rows(engine, 1) && fnfl_lane_owner(engine, 1) < 0;
        static int last_logged = -1;
        if (!fits && last_logged != slot.task->id) {
            last_logged = slot.task->id;
            LOG_WRN("TU pick: slot %d's next chunk (%d rows) exceeds the other-reader lane (%d rows): it waits for slot %d's "
                    "turn to end\n", slot.id, r.next_rows, fnfl_lane_rows(engine, 1), tu_job[0].slot ? tu_job[0].slot->id : -1);
        }
        return fits;
    }

    // (tickets TU1, TU2, TU6) one iteration: begin the started reads; order the readers; then a nested job's slice
    // (decode first), or the live job's slice or another reader's chunk out of turn (decode first), or the decode
    // steps and the pick's next turn. Returns to the queue loop between slices, so new and cancelled requests are seen
    // one line an iteration while a reader's chunk and at least one writer share it (the writers' service between
    // a job's slices); none for a lone reader or lone writers
    int64_t tu_iter = 0;
    void tu_iter_line(int k) {
        if (tu_stepped.empty()) {
            return;
        }
        const tu_job_t & j = tu_job[k];
        LOG_INF("TU iteration %lld: t_us=%" PRId64 " writers stepped [%s]; then job %d (slot %d, %s)\n",
                (long long) ++tu_iter, ggml_time_us(), tu_stepped.c_str(), k, j.slot ? j.slot->id : -1,
                j.active ? "slice" : "turn ended");
    }

    void tu_update() {
        // FMU1 (resident TP): who has model work this iteration
        int32_t n_writers = 0, n_reading = 0, n_started = 0;
        server_slot * writer = nullptr;
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_GENERATING) {
                ++n_writers;
                writer = &slot;
            }
            n_reading += slot.state == SLOT_STATE_PROCESSING_PROMPT ? 1 : 0;
            n_started += slot.state == SLOT_STATE_STARTED ? 1 : 0;
        }
        if (tu_resident && tu_lone_writer(n_writers, n_reading + n_started, tu_job[0].active || tu_job[1].active)) {
            // a lone writer: the one-slot loop's body (I2)
            try {
                metrics_on_step();
                step_slot(*writer);
            } catch (const std::exception & e) {
                SLT_ERR(*writer, "got exception: %s\n", e.what());
                send_error(*writer, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                writer->release();
            }
            return;
        }
        // admissions: FMU1, one an iteration while other slots have model work, and with writers present their
        // rounds follow it before any slice (an admission's truncate, restore and page mapping are a slice's work)
        const bool busy = n_writers > 0 || n_reading > 0;
        int32_t n_admitted = 0;
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_STARTED) {
                if (tu_resident && !tu_admit_more(busy, n_admitted)) {
                    break;
                }
                ++n_admitted;
                tu_admit(slot);
            }
        }
        // a cancelled read with no chunk in progress stops now (BG1: no chunk started for it)
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT && slot.tu_cancel &&
                !(tu_job[0].active && tu_job[0].slot == &slot) && !(tu_job[1].active && tu_job[1].slot == &slot)) {
                fnfl_read_report r;
                if (fnfl_read_info(engine, slot.id, r) && !r.in_chunk) {
                    const fnfl_read_status st = fnfl_read_step(engine, slot.id, 1);
                    if (st == FNFL_READ_CANCELLED) {
                        tu_cancelled(slot);
                    } else if (st == FNFL_READ_ERROR) {
                        engine_failed(slot, FNFL_ERROR);
                    }
                }
            }
        }
        tu_order();
        if (tu_resident) {
            tu_prefetch();
            if (busy && n_admitted > 0 && n_writers > 0) {
                tu_decode();
                return;
            }
        }
        if (tu_job[1].active) {
            tu_decode();
            tu_iter_line(1);
            tu_slice(1);
            return;
        }
        if (tu_job[0].active) {
            tu_decode();
            if (!tu_job[0].active) {
                return;
            }
            if (tu_pick != nullptr && tu_pick != tu_job[0].slot && tu_other_fits(*tu_pick)) {
                tu_iter_line(0);
                tu_start(*tu_pick, 1);
                return;
            }
            tu_iter_line(0);
            tu_slice(0);
            return;
        }
        tu_decode();
        if (tu_pick != nullptr && tu_pick->state == SLOT_STATE_PROCESSING_PROMPT) {
            tu_start(*tu_pick, 0);
        }
    }

    void update_slots() {
        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");
                return; // skip further processing
            }

            SRV_DBG("%s", "posting NEXT_RESPONSE\n");

            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(task));
        }

        // (MU1) more than one slot: two-user scheduling (the one-slot loop below is unchanged)
        if (tu_engaged) {
            tu_update();
            return;
        }

        for (auto & slot : slots) {
            try {
                if (slot.state == SLOT_STATE_STARTED) {
                    start_prompt(slot);
                } else if (slot.state == SLOT_STATE_GENERATING) {
                    metrics_on_step();
                    step_slot(slot);
                }
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // context size of a single slot: the engine's, capped by --kv-unified-per-slot
    int n_ctx_slot() const {
        return n_ctx_slot_val;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    //
    // metrics helpers
    //

    void metrics_on_step() {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->vocab, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->vocab, eos_id, true) : "";

    const auto & mm = impl->model_meta;

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->n_ctx_slot(),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ mm.n_ctx_train,
        /* model_n_embd_inp       */ mm.n_embd,
        /* model_n_params         */ mm.n_params,
        /* model_size             */ mm.size,
        /* model_ftype            */ mm.format_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        inputs = tokenize_input_prompts(ctx_server.vocab, prompt, true, true);

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        json delims = json_value(data, "message_delimiters", json::array());
        auto delimiters = common_chat_msg_delimiters_parse(delims);
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping,
                          const json & tu = json::object()) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", false},
            {"video",  false},
            {"audio",  false},
        } },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
        { "tu",                          tu }, // (ticket TU2; MU1) two-user scheduling, the effective values
    };
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save" || action == "restore") {
            res->error(format_error_response("slot " + action + " is not supported by FNFL", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false, ctx_server.tu_props_json()));
        }
        return res;
    };

    this->post_props = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        // (ticket TU2; MU1) two-user scheduling: {"tu": {"slice_ms": 10..10000, "group": layers}}. Validated here, and a
        // rejected request changes nothing; the owner thread applies it between steps and answers with the effective
        // values. FNFL's TU is engaged whenever the server has more than one slot, and TU6's fix and fair_read are its
        // permanent behavior: every other tu. field is rejected
        json body;
        try {
            body = req.body.empty() ? json::object() : json::parse(req.body);
        } catch (const std::exception & e) {
            res->error(format_error_response(std::string("invalid JSON: ") + e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (body.is_object() && body.contains("tu")) {
            const json & tu = body.at("tu");
            server_task task(SERVER_TASK_TYPE_TU_POLICY);
            std::string err;
            if (!tu.is_object()) {
                err = "\"tu\" must be an object";
            }
            for (auto it = tu.begin(); err.empty() && it != tu.end(); ++it) {
                const std::string & key = it.key();
                const json & v = it.value();
                if (key == "slice_ms") {
                    if (!v.is_number_integer() || v.get<int64_t>() < 10 || v.get<int64_t>() > 10000) {
                        err = "tu.slice_ms must be an integer from 10 to 10000";
                    } else {
                        task.tu_set.slice_ms = v.get<int32_t>();
                    }
                } else if (key == "group") {
                    if (!v.is_number_integer() || v.get<int64_t>() < 1) {
                        err = "tu.group must be a positive integer (layers per group)";
                    } else {
                        task.tu_set.group = (int32_t) std::min<int64_t>(v.get<int64_t>(), INT32_MAX);
                    }
                } else if (key == "enabled" || key == "fix" || key == "fair_read") {
                    err = "tu." + key + " is not a setting on FNFL: two-user scheduling is engaged with more than one "
                          "slot, and TU6's fix and fair read are its permanent behavior";
                } else if (key == "force" || key == "diag" || key == "peer_rows" || key == "version") {
                    err = "tu." + key + " is not a setting on FNFL";
                } else {
                    err = "unknown field tu." + key;
                }
            }
            if (!err.empty()) {
                res->error(format_error_response(err, ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            auto & rd = res->rd;
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
            auto result = rd.next(req.should_stop);
            if (!result) {
                GGML_ASSERT(req.should_stop());
                return res;
            }
            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }
            res->ok(result->to_json());
            return res;
        }

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    const size_t n_tokens = tokenize_mixed(vocab, prompt, true, true).size();

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true, ctx_server.tu_props_json());
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
