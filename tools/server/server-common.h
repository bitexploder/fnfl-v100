#pragma once

#include "common.h"
#include "log.h"
#include "llama.h"
#include "chat.h"
#include "subproc.h"

#include "json.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cinttypes>
#include <cstdio>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

using json = common_json;

#define SLT_DBG(slot, fmt, ...) LOG_DBG("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_TRC(slot, fmt, ...) LOG_TRC("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_INF(slot, fmt, ...) LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...) LOG_WRN("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...) LOG_ERR("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_CNT(slot, fmt, ...) LOG_CNT(""                                 fmt,                                                                __VA_ARGS__)

#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_TRC(fmt, ...) LOG_TRC("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

using raw_buffer = std::vector<uint8_t>;

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key).get<T>();
        } catch (const common_json_error & err) {
            LOG_WRN("Wrong type supplied for parameter '%s', using default value: %s\n", key.c_str(), err.what());
            return default_value;
        }
    } else {
        return default_value;
    }
}

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
    ERROR_TYPE_EXCEED_CONTEXT_SIZE, // custom error
};

// thin wrapper around common_grammar_trigger with (de)serialization functions
struct server_grammar_trigger {
    common_grammar_trigger value;

    server_grammar_trigger() = default;
    server_grammar_trigger(const common_grammar_trigger & value) : value(value) {}
    server_grammar_trigger(const json & in) {
        value.type = (common_grammar_trigger_type) in.at("type").get<int>();
        value.value = in.at("value").get<std::string>();
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            value.token = (llama_token) in.at("token").get<int>();
        }
    }

    json to_json() const {
        json out {
            {"type", (int) value.type},
            {"value", value.value},
        };
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            out["token"] = (int) value.token;
        }
        return out;
    }
};

json format_error_response(const std::string & message, const enum error_type type);

//
// random string / id
//

std::string random_string();
std::string gen_chatcmplid();
std::string gen_tool_call_id();

//
// server_tokens
//

// the tokens of a prompt or of a slot's engine state; FNFL serves text only
struct server_tokens {
private:
    llama_tokens tokens;

public:
    server_tokens() = default;
    ~server_tokens() = default;

    // copies are explicit (clone)
    server_tokens(const server_tokens &) = delete;
    server_tokens & operator=(const server_tokens &) = delete;

    server_tokens(server_tokens &&) = default;
    server_tokens & operator=(server_tokens &&) = default;

    llama_token operator[](size_t index) { return tokens[index]; }
    const llama_token & operator[](size_t index) const { return tokens[index]; }

    explicit server_tokens(const llama_tokens & tokens);

    // for debugging
    std::string str() const;

    void push_back(llama_token tok);
    void push_back(const server_tokens & tokens);

    void insert(const llama_tokens & inp_tokens);

    const llama_tokens & get_tokens() const;

    size_t size() const { return tokens.size(); }

    bool empty() const { return tokens.empty(); }

    void clear() { tokens.clear(); }

    void keep_first(size_t n);

    std::string detokenize(const llama_vocab * vocab, bool special) const;

    size_t get_common_prefix(const server_tokens & b) const;

    common_chat_msg_spans find_message_spans(const common_chat_msg_delimiters & delims) const;

    // every token is in the vocabulary
    bool validate(const llama_vocab * vocab) const;

    server_tokens clone() const;
};


//
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json & data);

// is array having BOTH numbers & strings?
bool json_is_array_of_mixed_numbers_strings(const json & data);

// does array have any individual integers/tokens?
bool json_is_array_and_contains_numbers(const json & data);

// get value by path(key1 / key2)
json json_get_nested_values(const std::vector<std::string> & paths, const json & js);

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special);

// return the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if validate_utf8(text) == text.size(), then the whole text is valid utf8
size_t validate_utf8(const std::string& text);

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string" }
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56], { "prompt_string": "string" }]
 */
std::vector<server_tokens> tokenize_input_prompts(
                                        const llama_vocab * vocab,
                                        const json & json_prompt,
                                        bool add_special,
                                        bool parse_special);

//
// OAI utils
//

// global server parameters for chat formatting / parsing
struct server_chat_params {
    bool use_jinja;
    bool prefill_assistant;
    common_reasoning_format reasoning_format;
    std::map<std::string, std::string> chat_template_kwargs; // mapping key --> json value
    common_chat_templates_ptr tmpls;
    bool enable_thinking = true;
    int  reasoning_budget = -1;
    std::string reasoning_budget_message;
    bool force_pure_content = false;
};

// used by /completions endpoint
json oaicompat_completion_params_parse(const json & body);

// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    json & body, /* openai api json semantics */
    const server_chat_params & opt);

//
// stats and metrics
//

// shared between server_slot and server_task_result_*
struct server_slot_stats {
    uint64_t n_prompt_cached    = 0;
    uint64_t n_prompt_processed = 0;
    uint64_t n_gen              = 0;

    // speculative decoding stats
    // note: the per-position breakdown lives in server_slot, it is not needed in a task result
    uint64_t n_draft_tokens      = 0;
    uint64_t n_draft_accepted    = 0;
    uint64_t n_draft_verif_steps = 0;

    // these are absolute timestamps (in us)
    // note: must be signed - they are subtracted before the later ones are set
    int64_t t_start       = 0;
    int64_t t_prompt_last = 0;
    int64_t t_gen_last    = 0;

    // can only move one direction: start -> prompt -> gen
    void update_prompt_start() {
        GGML_ASSERT(t_start == 0);
        t_start = ggml_time_us();
    }
    void set_prompt_last(int64_t t_us) {
        GGML_ASSERT(t_start > 0);
        t_prompt_last = t_us;
    }
    void update_prompt_last() {
        set_prompt_last(ggml_time_us());
    }
    void update_gen_last() {
        GGML_ASSERT(t_prompt_last > 0);
        t_gen_last = ggml_time_us();
    }

    // these are time durations
    int64_t t_elapsed_us() const {
        return ggml_time_us() - t_start;
    }
    double t_prompt_ms() const {
        if (t_prompt_last == 0) {
            return 0.0; // the prompt is not processed yet
        }
        return (t_prompt_last - t_start) / 1000.0;
    }
    int64_t t_gen_us() const {
        if (t_gen_last == 0) {
            return 0; // the generation is not started yet
        }
        // clamp to 1 us, the first token can land in the same us as t_prompt_last
        return std::max<int64_t>(1, t_gen_last - t_prompt_last);
    }
    double t_gen_ms() const {
        return t_gen_us() / 1000.0;
    }

    // number of decode steps spent on generation
    // the first token is free, it comes from the logits of the last prompt batch
    uint64_t n_gen_steps() const {
        return n_gen > 0 ? n_gen - 1 : 0;
    }

    // other derived metrics
    // note: all of them return 0.0 if the divisor is not known yet
    double t_prompt_per_token_ms() const {
        return n_prompt_processed > 0 ? t_prompt_ms() / n_prompt_processed : 0.0;
    }
    double t_gen_per_token_ms() const {
        return n_gen_steps() > 0 ? t_gen_ms() / n_gen_steps() : 0.0;
    }
    double n_prompt_tps() const {
        const double t_ms = t_prompt_ms();
        return t_ms > 0.0 ? 1e3 / t_ms * n_prompt_processed : 0.0;
    }
    double n_gen_tps() const {
        const double t_ms = t_gen_ms();
        return t_ms > 0.0 ? 1e3 / t_ms * n_gen_steps() : 0.0;
    }

    // false if the slot never started, i.e. the task result carries no stats
    bool is_set() const {
        return t_start > 0;
    }

    json to_json() const;
};

// shared between server_context_impl and server_task_result_*
// unlike server_slot_stats, server_metrics is server-global and cumulative, not tied to a slot
struct server_metrics {
    int64_t t_start = 0;

    struct bucket {
        uint64_t count = 0; // number of tokens
        uint64_t steps = 0; // number of decode steps,
                            // this excludes first generated token (logits from prompt batch)
        uint64_t time  = 0; // in microseconds

        // the rate uses the decode steps, so that "free" tokens do not inflate it
        double n_per_second() const {
            return time > 0 ? (double) steps / (double) time * 1e6 : 0.0;
        }

        void add(uint64_t n, uint64_t n_steps, uint64_t t_us) {
            count += n;
            steps += n_steps;
            time  += t_us;
        }
    };

    // these are reset by reset_bucket(), only the rate is read from them
    bucket prompt_bucket;
    bucket predict_bucket;

    // metrics below are cumulative since the server started
    bucket prompt; // only processed tokens, cached ones are counted separately below
    bucket predict;

    // tokens reused from the cache need no decode, so they only have a count
    uint64_t n_prompt_cached = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_decode     = 0;
    uint64_t n_busy_slots = 0;

    uint64_t n_draft_tokens      = 0; // Total draft tokens generated
    uint64_t n_draft_accepted    = 0; // Draft tokens actually accepted
    uint64_t n_draft_verif_steps = 0; // Total draft token verification steps by the target model
    std::vector<uint64_t> n_accepted_per_pos; // Accepted tokens per draft position

    void init() {
        t_start = ggml_time_us();
    }

    void reset_bucket() {
        prompt_bucket  = {};
        predict_bucket = {};
    }

    void add_prompt(uint64_t n_tokens, uint64_t t_us) {
        prompt       .add(n_tokens, n_tokens, t_us);
        prompt_bucket.add(n_tokens, n_tokens, t_us);
    }

    void add_prompt_cached(uint64_t n_tokens) {
        n_prompt_cached += n_tokens;
    }
};

//
// other utils
//


std::string safe_json_to_str(const json & data);

std::string tokens_to_str(const llama_vocab * vocab, const llama_tokens & tokens);

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string format_oai_sse(const json & data);

std::string format_oai_resp_sse(const json & data);

// format Anthropic-style SSE with event types
std::string format_anthropic_sse(const json & data);

bool is_valid_utf8(const std::string & str);

//
// formatting output responses
// TODO: move these to server-task.cpp
//

llama_tokens format_prompt_infill(
        const llama_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const llama_tokens & tokens_prompt);

// simple implementation of a pipe
// used for streaming data between threads
template<typename T>
struct server_pipe {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<T> queue;
    std::atomic<bool> writer_closed{false};
    std::atomic<bool> reader_closed{false};

    // 0 = unbounded (default)
    // > 0, write() drops the oldest item once the queue is full
    size_t max_size = 0;

    void close_write() {
        writer_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }

    void close_read() {
        reader_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }

    // close_on_stop = true: should_stop means the reader is gone for good, so the writer is told the pipe is broken.
    // close_on_stop = false: should_stop is a per-read deadline and further reads still come, so the pipe stays usable.
    bool read(T & output, const std::function<bool()> & should_stop, bool close_on_stop = true) {
        std::unique_lock<std::mutex> lk(mutex);
        constexpr auto poll_interval = std::chrono::milliseconds(500);
        while (true) {
            if (!queue.empty()) {
                output = std::move(queue.front());
                queue.pop();
                return true;
            }
            if (writer_closed.load()) {
                return false; // clean EOF
            }
            if (should_stop && should_stop()) { // a null should_stop means "never stop"
                if (close_on_stop) {
                    close_read(); // signal broken pipe to writer
                }
                return false; // cancelled / deadline reached
            }
            cv.wait_for(lk, poll_interval);
        }
    }

    bool write(T && data) {
        std::lock_guard<std::mutex> lk(mutex);
        if (reader_closed.load()) {
            return false; // broken pipe
        }
        if (max_size > 0) {
            while (queue.size() >= max_size) {
                queue.pop(); // drop oldest to stay bounded
            }
        }
        queue.push(std::move(data));
        cv.notify_one();
        return true;
    }
};

// wrapper around common_subproc to manage a child server process
// mainly used by router mode
struct server_subproc {
    common_subproc sproc;
    std::atomic<bool> stopped{false}; // set by the monitor once the process exited and was reaped

    bool is_alive() { return sproc.alive(); }
    void terminate() { sproc.terminate(); }
    int  join() { return sproc.join(); }

    // true if the child's combined stdout/stderr pipe is available (call after create())
    bool has_output();

    // non-blocking read
    // returns the number of bytes read, 0 when nothing is available, -1 when the pipe is closed or broken
    int read_output(char * buf, size_t len);

    // wait until one of a set of children has output, wake() is called, or a timeout passes
    struct waiter {
        waiter();
        ~waiter();

        // thread-safe; on Windows this is a no-op, wait() returns within 50 ms anyway
        void wake();

        // timeout_ms < 0 waits until data or wake(); ready[i] is set for each child with data (or a broken pipe)
        void wait(const std::vector<server_subproc *> & procs, std::vector<bool> & ready, int64_t timeout_ms);

    private:
        intptr_t wake_fd[2] = { -1, -1 }; // POSIX self-pipe
    };

private:
    intptr_t out_handle = -1; // fd on POSIX, HANDLE on Windows; taken lazily from sproc
};
