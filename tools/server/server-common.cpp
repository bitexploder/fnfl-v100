#include "common.h"
#include "download.h"
#include "log.h"
#include "llama.h"
#include "chat.h"
#include "base64.hpp"

#include "server-common.h"

#include <random>
#include <sstream>
#include <fstream>
#include <limits>
#include <cstring>
#include <type_traits>
#include <chrono>
#include <thread>

#ifdef _WIN32
// windows.h defines min and max as macros, which breaks std::min and std::max
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#endif

json format_error_response(const std::string & message, const enum error_type type) {
    std::string type_str;
    int code = 500;
    switch (type) {
        case ERROR_TYPE_INVALID_REQUEST:
            type_str = "invalid_request_error";
            code = 400;
            break;
        case ERROR_TYPE_AUTHENTICATION:
            type_str = "authentication_error";
            code = 401;
            break;
        case ERROR_TYPE_NOT_FOUND:
            type_str = "not_found_error";
            code = 404;
            break;
        case ERROR_TYPE_SERVER:
            type_str = "server_error";
            code = 500;
            break;
        case ERROR_TYPE_PERMISSION:
            type_str = "permission_error";
            code = 403;
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            type_str = "not_supported_error";
            code = 501;
            break;
        case ERROR_TYPE_UNAVAILABLE:
            type_str = "unavailable_error";
            code = 503;
            break;
        case ERROR_TYPE_EXCEED_CONTEXT_SIZE:
            type_str = "exceed_context_size_error";
            code = 400;
            break;
    }
    return json {
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}

//
// server_slot_stats
//

json server_slot_stats::to_json() const {
    json base = {
        {"cache_n",                n_prompt_cached},

        {"prompt_n",               n_prompt_processed},
        {"prompt_ms",              t_prompt_ms()},
        {"prompt_per_token_ms",    t_prompt_per_token_ms()},
        {"prompt_per_second",      n_prompt_tps()},

        {"predicted_n",            n_gen},
        {"predicted_ms",           t_gen_ms()},
        {"predicted_per_token_ms", t_gen_per_token_ms()},
        {"predicted_per_second",   n_gen_tps()},
    };

    if (n_draft_tokens > 0) {
        base["draft_n"]          = n_draft_tokens;
        base["draft_n_accepted"] = n_draft_accepted;
    }

    return base;
}

//
// random string / id
//

std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

std::string gen_chatcmplid() {
    return "chatcmpl-" + random_string();
}

std::string gen_tool_call_id() {
    return random_string();
}


//
// server_tokens implementation
//

server_tokens::server_tokens(const llama_tokens & tokens) : tokens(tokens) {
}

std::string server_tokens::str() const {
    std::ostringstream oss;
    oss << "tokens: ";
    for (size_t idx = 0; idx < tokens.size(); ++idx) {
        oss << "idx:" << idx << " " << tokens[idx] << " ";
    }
    return oss.str();
}

void server_tokens::push_back(llama_token tok) {
    if (tok == LLAMA_TOKEN_NULL) {
        throw std::runtime_error("Invalid token");
    }
    tokens.emplace_back(tok);
}

void server_tokens::push_back(const server_tokens & other) {
    for (size_t i = 0; i < other.size(); i++) {
        push_back(other[i]);
    }
}

void server_tokens::insert(const llama_tokens & inp_tokens) {
    tokens.insert(tokens.end(), inp_tokens.begin(), inp_tokens.end());
}

const llama_tokens & server_tokens::get_tokens() const {
    return tokens;
}

void server_tokens::keep_first(size_t n) {
    GGML_ASSERT(n <= tokens.size());
    tokens.resize(n);
}

std::string server_tokens::detokenize(const llama_vocab * vocab, bool special) const {
    return common_detokenize(vocab, tokens, special);
}

size_t server_tokens::get_common_prefix(const server_tokens & b) const {
    const size_t max_idx = std::min(tokens.size(), b.tokens.size());

    for (size_t i = 0; i < max_idx; ++i) {
        if (tokens[i] != b.tokens[i]) {
            return i;
        }
    }

    return max_idx;
}

common_chat_msg_spans server_tokens::find_message_spans(const common_chat_msg_delimiters & delims) const {
    return delims.split(tokens, {});
}

bool server_tokens::validate(const llama_vocab * vocab) const {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (const llama_token t : tokens) {
        if (t < 0 || t >= n_vocab) {
            return false;
        }
    }
    return true;
}

server_tokens server_tokens::clone() const {
    return server_tokens(tokens);
}

//
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (!e.is_number_integer()) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool json_is_array_of_mixed_numbers_strings(const json & data) {
    bool seen_string = false;
    bool seen_number = false;
    if (data.is_array()) {
        for (const auto & e : data) {
            seen_string |= e.is_string();
            seen_number |= e.is_number_integer();
            if (seen_number && seen_string) {
                return true;
            }
        }
    }
    return false;
}

bool json_is_array_and_contains_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (e.is_number_integer()) {
                return true;
            }
        }
        return false;
    }
    return false;
}

json json_get_nested_values(const std::vector<std::string> & paths, const json & js) {
    json result = json::object();

    for (const std::string & path : paths) {
        json current = js;
        const auto keys = string_split<std::string>(path, /*separator*/ '/');
        bool valid_path = true;
        for (const std::string & k : keys) {
            if (valid_path && current.is_object() && current.contains(k)) {
                current = current[k];
            } else {
                valid_path = false;
            }
        }
        if (valid_path) {
            result[path] = current;
        }
    }
    return result;
}

llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    llama_tokens prompt_tokens;

    if (json_prompt.is_array()) {
        bool first = true;
        for (const auto & p : json_prompt) {
            if (p.is_string()) {
                auto s = p.template get<std::string>();

                llama_tokens p;
                if (first) {
                    p = common_tokenize(vocab, s, add_special, parse_special);
                    first = false;
                } else {
                    p = common_tokenize(vocab, s, false, parse_special);
                }

                prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
            } else {
                if (first) {
                    first = false;
                }

                prompt_tokens.push_back(p.template get<llama_token>());
            }
        }
    } else {
        auto s = json_prompt.template get<std::string>();
        prompt_tokens = common_tokenize(vocab, s, add_special, parse_special);
    }

    return prompt_tokens;
}

size_t validate_utf8(const std::string& text) {
    size_t len = text.size();
    if (len == 0) return 0;

    // Check the last few bytes to see if a multi-byte character is cut off
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = text[len - i];
        // Check for start of a multi-byte sequence from the end
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character start: 110xxxxx
            // Needs at least 2 bytes
            if (i < 2) return len - i;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character start: 1110xxxx
            // Needs at least 3 bytes
            if (i < 3) return len - i;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character start: 11110xxx
            // Needs at least 4 bytes
            if (i < 4) return len - i;
        }
    }

    // If no cut-off multi-byte character is found, return full length
    return len;
}

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * use tokenize_input_prompts() if the input could be an array.
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string" }
 */
static server_tokens tokenize_input_subprompt(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    constexpr char JSON_STRING_PROMPT_KEY[] = "prompt_string";
    constexpr char JSON_MTMD_DATA_KEY[] = "multimodal_data";
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt)) {
        // string or mixed
        llama_tokens tmp = tokenize_mixed(vocab, json_prompt, add_special, parse_special);
        return server_tokens(tmp);
    } else if (json_is_array_of_numbers(json_prompt)) {
        // array of tokens
        llama_tokens tmp = json_prompt.get<llama_tokens>();
        return server_tokens(tmp);
    } else if (json_prompt.contains(JSON_STRING_PROMPT_KEY)) {
        // JSON object with prompt key.
        if (json_prompt.contains(JSON_MTMD_DATA_KEY)) {
            throw std::runtime_error("Multimodal data provided, but FNFL serves text only.");
        }
        llama_tokens tmp = tokenize_mixed(vocab, json_prompt.at(JSON_STRING_PROMPT_KEY), add_special, parse_special);
        return server_tokens(tmp);
   } else {
       throw std::runtime_error("\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens.");
   }
}

std::vector<server_tokens> tokenize_input_prompts(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    std::vector<server_tokens> result;
    if (json_prompt.is_array() && !json_is_array_and_contains_numbers(json_prompt)) {
        result.reserve(json_prompt.size());
        for (const auto & p : json_prompt) {
            result.push_back(tokenize_input_subprompt(vocab, p, add_special, parse_special));
        }
    } else {
        result.push_back(tokenize_input_subprompt(vocab, json_prompt, add_special, parse_special));
    }
    if (result.empty()) {
        throw std::runtime_error("\"prompt\" must not be empty");
    }
    return result;
}

//
// OAI utils
//

// used by /completions endpoint
json oaicompat_completion_params_parse(const json & body) {
    json llama_params;

    if (!body.contains("prompt")) {
        throw std::runtime_error("\"prompt\" is required");
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    // Handle "echo" field
    if (json_value(body, "echo", false)) {
        throw std::runtime_error("Only no echo is supported");
    }

    // Params supported by OAI but unsupported by llama.cpp
    static const std::vector<std::string> unsupported_params { "best_of", "suffix" };
    for (const auto & param : unsupported_params) {
        if (body.contains(param)) {
            throw std::runtime_error("Unsupported param: " + param);
        }
    }

    // Copy remaining properties to llama_params
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    json & body, /* openai api json semantics */
    const server_chat_params & opt)
{
    json llama_params;

    auto tools = json_value(body, "tools", json());
    auto has_tools = tools.is_array() && !tools.empty();
    auto stream = json_value(body, "stream", false);
    auto tool_choice = json_value(body, "tool_choice", std::string("auto"));

    if (!opt.use_jinja) {
        if (has_tools) {
            throw std::runtime_error("tools param requires --jinja flag");
        }
        if (tool_choice != "auto") {
            throw std::runtime_error("tool_choice param requires --jinja flag");
        }
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    auto json_schema = json_value(body, "json_schema", json());
    auto grammar = json_value(body, "grammar", std::string());
    if (!json_schema.is_null() && !grammar.empty()) {
        throw std::runtime_error("Cannot use both json_schema and grammar");
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format      = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            if (response_format.contains("schema") || json_schema.empty()) {
                json_schema = json_value(response_format, "schema", json::object());
            }
        } else if (response_type == "json_schema") {
            auto schema_wrapper = json_value(response_format, "json_schema", json::object());
            json_schema = json_value(schema_wrapper, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::invalid_argument("response_format type must be one of \"text\" or \"json_object\", but got: " + response_type);
        }
    }

    // FNFL: constrained output is refused until the operator decides whether FNFL supports it
    if (!json_schema.is_null() || !grammar.empty()) {
        throw std::invalid_argument("grammar and JSON-schema constrained output are not supported by FNFL");
    }

    // an absent or empty schema means any object
    if (json_schema.is_object() && json_schema.empty()) {
        json_schema["type"] = "object";
    }

    // get input files
    if (!body.contains("messages")) {
        throw std::invalid_argument("'messages' is required");
    }
    json & messages = body.at("messages");
    if (!messages.is_array()) {
        throw std::invalid_argument("Expected 'messages' to be an array");
    }
    for (auto & msg : messages) {
        std::string role = json_value(msg, "role", std::string());
        if (role != "assistant" && !msg.contains("content")) {
            throw std::invalid_argument("All non-assistant messages must contain 'content'");
        }
        if (role == "assistant") {
            if (!msg.contains("content") && !msg.contains("tool_calls")) {
                throw std::invalid_argument("Assistant message must contain either 'content' or 'tool_calls'!");
            }
            if (!msg.contains("content")) {
                continue; // avoid errors with no content
            }
        }
        json & content = msg.at("content");
        if (content.is_string() || content.is_null()) {
            continue;
        }

        if (!content.is_array()) {
            throw std::invalid_argument("Expected 'content' to be a string or an array");
        }

        for (auto & p : content) {
            std::string type = json_value(p, "type", std::string());
            if (type == "image_url" || type == "input_audio" || type == "input_video") {
                throw std::runtime_error(type + " input is not supported: FNFL serves text only");
            } else if (type != "text") {
                throw std::invalid_argument("unsupported content[].type");
            }
        }
    }

    auto caps = common_chat_templates_get_caps(opt.tmpls.get());

    common_chat_templates_inputs inputs;
    inputs.messages               = common_chat_msgs_parse_oaicompat(messages);
    inputs.tools                  = common_chat_tools_parse_oaicompat(tools);
    inputs.tool_choice            = common_chat_tool_choice_parse_oaicompat(tool_choice);
    inputs.json_schema            = json_schema.is_null() ? "" : json_schema.dump();
    inputs.grammar                = grammar;
    inputs.use_jinja              = opt.use_jinja;
    inputs.parallel_tool_calls    = json_value(body, "parallel_tool_calls", caps["supports_parallel_tool_calls"]);
    inputs.add_generation_prompt  = json_value(body, "add_generation_prompt", true);
    inputs.continue_final_message = body.contains("continue_final_message") ?
        common_chat_continuation_parse(body.at("continue_final_message")) :
        COMMON_CHAT_CONTINUATION_NONE;
    if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_NONE && opt.prefill_assistant
        && !inputs.messages.empty() && inputs.messages.back().role == "assistant") {
        if (inputs.messages.size() >= 2 && inputs.messages[inputs.messages.size() - 2].role == "assistant") {
            throw std::invalid_argument("Cannot have 2 or more assistant messages at the end of the list.");
        }
        inputs.continue_final_message = COMMON_CHAT_CONTINUATION_AUTO;
        inputs.add_generation_prompt  = false;
    }
    if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && inputs.add_generation_prompt) {
        throw std::invalid_argument("Cannot set both add_generation_prompt and continue_final_message to true.");
    }
    if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE
        && !inputs.messages.empty()
        && inputs.messages.back().role == "assistant"
        && !inputs.messages.back().tool_calls.empty()) {
        throw std::invalid_argument("Cannot continue an assistant message that contains tool calls.");
    }
    inputs.reasoning_format = opt.reasoning_format;
    if (body.contains("reasoning_format")) {
        inputs.reasoning_format = common_reasoning_format_from_name(body.at("reasoning_format").get<std::string>());
    }
    inputs.enable_thinking = opt.enable_thinking;
    if (!inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
        if (body.contains("grammar")) {
            throw std::invalid_argument("Cannot use custom grammar constraints with tools.");
        }
        llama_params["parse_tool_calls"] = true;
    }

    // merge the template args provided from command line with the args provided in the user request
    auto chat_template_kwargs_object = json_value(body, "chat_template_kwargs", json::object());
    inputs.chat_template_kwargs = opt.chat_template_kwargs;
    for (const auto & item : chat_template_kwargs_object.items()) {
        inputs.chat_template_kwargs[item.key()] = item.value().dump();
    }

    // parse the "enable_thinking" kwarg to override the default value
    auto enable_thinking_kwarg = json_value(inputs.chat_template_kwargs, "enable_thinking", std::string(""));
    if (enable_thinking_kwarg == "true") {
        inputs.enable_thinking = true;
    } else if (enable_thinking_kwarg == "false") {
        inputs.enable_thinking = false;
    } else if (!enable_thinking_kwarg.empty() && enable_thinking_kwarg[0] == '"') {
        throw std::invalid_argument("invalid type for \"enable_thinking\" (expected boolean, got string)");
    }

    // Parse the OAI "reasoning_effort" field; "none" disables reasoning.
    if (body.contains("reasoning_effort")) {
        auto reasoning_effort = json_value(body, "reasoning_effort", std::string(""));
        if (reasoning_effort == "none") {
            inputs.enable_thinking = false;
            inputs.chat_template_kwargs.erase("reasoning_effort");
        } else if (!reasoning_effort.empty()) {
            inputs.chat_template_kwargs["reasoning_effort"] = json(reasoning_effort).dump();
        }
    }

    inputs.force_pure_content = opt.force_pure_content;

    // Apply chat template to the list of messages
    auto chat_params = common_chat_templates_apply(opt.tmpls.get(), inputs);

    llama_params["chat_format"] = static_cast<int>(chat_params.format);
    llama_params["prompt"]      = chat_params.prompt;
    if (!chat_params.grammar.empty()) {
        llama_params["grammar"]      = chat_params.grammar;
        llama_params["grammar_type"] = std::string("tool_calls");
    }
    llama_params["grammar_lazy"] = chat_params.grammar_lazy;
    auto grammar_triggers        = json::array();
    for (const auto & trigger : chat_params.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }
    llama_params["grammar_triggers"]  = grammar_triggers;
    llama_params["preserved_tokens"]  = chat_params.preserved_tokens;
    llama_params["generation_prompt"] = chat_params.generation_prompt;
    for (const auto & stop : chat_params.additional_stops) {
        llama_params["stop"].push_back(stop);
    }
    if (!chat_params.parser.empty()) {
        llama_params["chat_parser"] = chat_params.parser;
    }

    llama_params["message_delimiters"] = chat_params.message_delimiters.to_json();

    // Reasoning budget: pass parameters through to sampling layer
    {
        int reasoning_budget = json_value(body, "reasoning_budget_tokens",
                               json_value(body, "thinking_budget_tokens", -1));
        if (reasoning_budget == -1) {
            reasoning_budget = opt.reasoning_budget;
        }

        if (!chat_params.thinking_end_tags.empty()) {
            llama_params["reasoning_budget_tokens"] = reasoning_budget;
            llama_params["reasoning_budget_start_tag"] = chat_params.thinking_start_tag;
            llama_params["reasoning_budget_end_tags"] = chat_params.thinking_end_tags;
            llama_params["reasoning_budget_message"] = json_value(body, "reasoning_budget_message", opt.reasoning_budget_message);
            llama_params["reasoning_control"] = json_value(body, "reasoning_control", false);
        }
    }

    // Handle "logprobs" field
    // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may need to fix it in the future
    if (json_value(body, "logprobs", false)) {
        if (has_tools && stream) {
            throw std::invalid_argument("logprobs is not supported with tools + stream");
        }
        llama_params["n_probs"] = json_value(body, "top_logprobs", 20);
    } else if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        throw std::invalid_argument("top_logprobs requires logprobs to be set to true");
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", ... via OAI endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

//
// other utils
//

std::string safe_json_to_str(const json & data) {
    return data.dump_safe();
}

// TODO: reuse llama_detokenize
template <class Iter>
static std::string tokens_to_str(const llama_vocab * ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += common_token_to_piece(ctx, *begin);
    }

    return ret;
}

std::string tokens_to_str(const llama_vocab * vocab, const llama_tokens & tokens) {
    return tokens_to_str(vocab, tokens.begin(), tokens.end());
}

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string format_oai_sse(const json & data) {
    std::ostringstream ss;
    auto send_single = [&ss](const json & data) {
        ss << "data: " <<
            safe_json_to_str(data) <<
            "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).
    };

    if (data.is_array()) {
        for (const auto & item : data) {
            send_single(item);
        }
    } else {
        send_single(data);
    }

    return ss.str();
}

std::string format_oai_resp_sse(const json & data) {
    std::ostringstream ss;
    auto send_single = [&ss](const json & event_obj) {
        ss << "event: " << event_obj.at("event").get<std::string>() << "\n";
        ss << "data: " << safe_json_to_str(event_obj.at("data")) << "\n\n";
    };

    if (data.is_array()) {
        for (const auto & item : data) {
            send_single(item);
        }
    } else {
        send_single(data);
    }

    return ss.str();
}

std::string format_anthropic_sse(const json & data) {
    std::ostringstream ss;

    auto send_event = [&ss](const json & event_obj) {
        if (event_obj.contains("event") && event_obj.contains("data")) {
            ss << "event: " << event_obj.at("event").get<std::string>() << "\n";
            ss << "data: " << safe_json_to_str(event_obj.at("data")) << "\n\n";
        } else {
            ss << "data: " << safe_json_to_str(event_obj) << "\n\n";
        }
    };

    if (data.is_array()) {
        for (const auto & event : data) {
            send_event(event);
        }
    } else {
        send_event(data);
    }

    return ss.str();
}

bool is_valid_utf8(const std::string & str) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str.data());
    const unsigned char* end = bytes + str.length();

    while (bytes < end) {
        if (*bytes <= 0x7F) {
            // 1-byte sequence (0xxxxxxx)
            bytes++;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // 2-byte sequence (110xxxxx 10xxxxxx)
            if (end - bytes < 2 || (bytes[1] & 0xC0) != 0x80)
                return false;
            bytes += 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 3 || (bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80)
                return false;
            bytes += 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 4 || (bytes[1] & 0xC0) != 0x80 ||
                (bytes[2] & 0xC0) != 0x80 || (bytes[3] & 0xC0) != 0x80)
                return false;
            bytes += 4;
        } else {
            // Invalid UTF-8 lead byte
            return false;
        }
    }

    return true;
}

llama_tokens format_prompt_infill(
        const llama_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const llama_tokens & tokens_prompt
    ) {
    // TODO: optimize this block by reducing memory allocations and movement

    // use FIM repo-level pattern:
    // ref: https://arxiv.org/pdf/2409.12186
    //
    // [FIM_REP]myproject
    // [FIM_SEP]filename0
    // extra chunk 0
    // [FIM_SEP]filename1
    // extra chunk 1
    // ...
    // [FIM_SEP]filename
    // [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
    //
    llama_tokens extra_tokens;
    extra_tokens.reserve(n_ctx);

    auto tokens_prefix = tokenize_mixed(vocab, input_prefix, false, false);
    auto tokens_suffix = tokenize_mixed(vocab, input_suffix, false, false);

    if (llama_vocab_fim_rep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: make project name an input
        static const auto k_fim_repo = common_tokenize(vocab, "myproject\n", false, false);

        extra_tokens.push_back(llama_vocab_fim_rep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_repo.begin(), k_fim_repo.end());
    }
    for (const auto & chunk : input_extra) {
        // { "text": string, "filename": string }
        const std::string text     = json_value(chunk, "text",     std::string());
        const std::string filename = json_value(chunk, "filename", std::string("tmp"));

        if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
            const auto k_fim_file = common_tokenize(vocab, filename + "\n", false, false);

            extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
            extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
        } else {
            // chunk separator in binary form to avoid confusing the AI
            static const char k_chunk_prefix_str[] = {0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70, 0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a, 0x00};
            static const auto k_chunk_prefix_tokens = common_tokenize(vocab, k_chunk_prefix_str, false, false);

            extra_tokens.insert(extra_tokens.end(), k_chunk_prefix_tokens.begin(), k_chunk_prefix_tokens.end());
        }

        const auto chunk_tokens = common_tokenize(vocab, text, false, false);
        extra_tokens.insert(extra_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    }

    if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: current filename
        static const auto k_fim_file = common_tokenize(vocab, "filename\n", false, false);

        extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
    }

    // for now pick FIM context to fit in a batch (ratio prefix:suffix = 3:1, TODO: configurable?)
    const int n_prefix_take = std::min<int>(tokens_prefix.size(),                3*(n_batch/4));
    const int n_suffix_take = std::min<int>(tokens_suffix.size(), std::max<int>(0, (n_batch/4) - (2 + tokens_prompt.size())));

    SRV_DBG("n_prefix_take = %d, n_suffix_take = %d, total = %d\n", n_prefix_take, n_suffix_take, (n_prefix_take + n_suffix_take));

    // fill the rest of the context with extra chunks
    const int n_extra_take = std::min<int>(std::max<int>(0, n_ctx - (n_batch) - 2*n_predict), extra_tokens.size());

    tokens_prefix.erase(tokens_prefix.begin(), tokens_prefix.begin() + tokens_prefix.size() - n_prefix_take);
    tokens_suffix.resize(n_suffix_take);

    tokens_prefix.insert(tokens_prefix.begin(), llama_vocab_fim_pre(vocab));
    tokens_prefix.insert(tokens_prefix.end(),   tokens_prompt.begin(), tokens_prompt.end());
    tokens_suffix.insert(tokens_suffix.begin(), llama_vocab_fim_suf(vocab));

    auto embd_inp = spm_infill ? tokens_suffix : tokens_prefix;
    auto embd_end = spm_infill ? tokens_prefix : tokens_suffix;

    if (llama_vocab_get_add_bos(vocab)) {
        embd_inp.insert(embd_inp.begin(), llama_vocab_bos(vocab));
    }

    SRV_DBG("extra: n_ctx = %d, n_extra_take = %d, n_extra = %d\n", n_ctx, n_extra_take, (int) extra_tokens.size());

    // put the extra context before the FIM prefix
    embd_inp.insert(embd_inp.begin(), extra_tokens.end() - n_extra_take, extra_tokens.end());

    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());
    embd_inp.push_back(llama_vocab_fim_mid(vocab));

    return embd_inp;
}

//
// server_subproc
//

bool server_subproc::has_output() {
    if (out_handle >= 0) {
        return true;
    }
    FILE * f = sproc.stdout_file(); // combined stdout/stderr
    if (!f) {
        return false;
    }
#ifdef _WIN32
    HANDLE h = (HANDLE) _get_osfhandle(_fileno(f));
    if (h != INVALID_HANDLE_VALUE) {
        out_handle = (intptr_t) h;
    }
#else
    int fd = fileno(f);
    if (fd >= 0) {
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
        out_handle = fd;
    }
#endif
    return out_handle >= 0;
}

int server_subproc::read_output(char * buf, size_t len) {
    if (!has_output()) {
        return -1;
    }
#ifdef _WIN32
    HANDLE h     = (HANDLE) out_handle;
    DWORD  avail = 0;
    if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL)) {
        return -1; // pipe broken, child gone
    }
    if (avail == 0) {
        return 0;
    }
    DWORD to_read = avail < (DWORD) len ? avail : (DWORD) len;
    DWORD got     = 0;
    if (!ReadFile(h, buf, to_read, &got, NULL) || got == 0) {
        return -1;
    }
    return (int) got;
#else
    while (true) {
        ssize_t r = read((int) out_handle, buf, len);
        if (r > 0) {
            return (int) r;
        }
        if (r == 0) {
            return -1; // EOF
        }
        if (errno == EINTR) {
            continue;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return 0;
        }
        return -1;
    }
#endif
}

server_subproc::waiter::waiter() {
#ifndef _WIN32
    int fds[2];
    GGML_ASSERT(pipe(fds) == 0);
    for (int fd : fds) {
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
    }
    wake_fd[0] = fds[0];
    wake_fd[1] = fds[1];
#endif
}

server_subproc::waiter::~waiter() {
#ifndef _WIN32
    close((int) wake_fd[0]);
    close((int) wake_fd[1]);
#endif
}

void server_subproc::waiter::wake() {
#ifndef _WIN32
    char c = 1;
    (void) !write((int) wake_fd[1], &c, 1);
#endif
}

void server_subproc::waiter::wait(const std::vector<server_subproc *> & procs, std::vector<bool> & ready, int64_t timeout_ms) {
    ready.assign(procs.size(), false);
#ifdef _WIN32
    // no waitable wait exists for anonymous pipes, so poll them in 50 ms steps
    bool any = false;
    for (size_t i = 0; i < procs.size(); i++) {
        DWORD avail = 0;
        if (!procs[i]->has_output() || !PeekNamedPipe((HANDLE) procs[i]->out_handle, NULL, 0, NULL, &avail, NULL) || avail > 0) {
            ready[i] = true; // data or broken pipe, read_output() tells which
            any = true;
        }
    }
    if (!any) {
        int64_t step = timeout_ms < 0 ? 50 : std::min<int64_t>(timeout_ms, 50);
        std::this_thread::sleep_for(std::chrono::milliseconds(step));
    }
#else
    std::vector<pollfd> pfds;
    pfds.reserve(procs.size() + 1);
    pfds.push_back({ (int) wake_fd[0], POLLIN, 0 });
    for (auto * p : procs) {
        pfds.push_back({ p->has_output() ? (int) p->out_handle : -1, POLLIN, 0 }); // poll() skips negative fds
    }
    int timeout = timeout_ms < 0 ? -1 : (int) std::min<int64_t>(timeout_ms, std::numeric_limits<int>::max());
    int r = poll(pfds.data(), pfds.size(), timeout);
    if (r < 0 && errno != EINTR) {
        LOG_ERR("%s: poll() failed: %s\n", __func__, strerror(errno));
    }
    if (pfds[0].revents) {
        char buf[64];
        while (read((int) wake_fd[0], buf, sizeof(buf)) > 0) {}
    }
    for (size_t i = 0; i < procs.size(); i++) {
        ready[i] = pfds[i + 1].fd < 0 || pfds[i + 1].revents != 0;
    }
#endif
}
