#include "server-reasoning-check.h"

#include "server-chat.h"
#include "server-schema.h"
#include "server-task.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <utility>

namespace {

constexpr int32_t def_budget = 64; // the cases' server default (--reasoning-budget)
constexpr int64_t any        = INT64_MIN;

struct reply {
    int         status = 200;
    std::string error;
    int64_t     budget = any;
    int         format = -1;
    std::string prompt;
    std::string message; // FBH1 A2: the effective budget message
};

// the route's steps (server-context.cpp routes): post_anthropic_messages translates, the chat routes parse with
// oaicompat_chat_params_parse, an exception there reaching ex_wrapper (format_exception_response); then
// handle_completions_impl evaluates the schema, where any exception is the client's (400)
reply route(const reasoning_check_ctx & c, const common_params & params, const std::string & r, json body) {
    reply out;
    const auto fail = [&](const json & e) {
        out.status = json_value(e, "code", 500);
        out.error  = json_value(e, "message", std::string());
    };
    try {
        if (r == "anthropic") {
            body = server_chat_convert_anthropic_to_oai(body);
        }
        json data = body;
        if (r != "completion") {
            data       = oaicompat_chat_params_parse(body, c.opt);
            out.prompt = json_value(data, "prompt", std::string());
        }
        try {
            const task_params tp = server_schema::eval_llama_cmpl_schema(c.vocab, params, c.logit_bias_eog, data);
            out.budget           = tp.sampling.reasoning_budget_tokens;
            out.message          = tp.sampling.reasoning_budget_message;
            out.format           = static_cast<int>(tp.chat_parser_params.reasoning_format);
        } catch (const std::exception & e) {
            fail(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        }
    } catch (...) {
        fail(format_exception_response(std::current_exception()));
    }
    return out;
}

struct pcase {
    const char * name;
    const char * route; // chat, completion or anthropic
    const char * extra; // the fields added to the route's minimal body
    int          status;
    int64_t      budget;    // the effective budget (any: not checked)
    const char * has;       // a text the error must hold (null: none)
    int          format;    // the parser's reasoning format (-1: not checked)
    bool         templated; // the rendered prompt differs from the plain request's
};

const pcase cases[] = {
    // R1: reasoning_format
    { "R1 chat: reasoning_format \"bogus\"", "chat", R"({"reasoning_format": "bogus"})", 400, any,
      "\"deepseek-legacy\"", -1, false },
    { "R1 chat: reasoning_format 5", "chat", R"({"reasoning_format": 5})", 400, any, "must be a string", -1, false },
    { "R1 chat: reasoning_format null (the server's)", "chat", R"({"reasoning_format": null})", 200, def_budget,
      nullptr, -1, false },
    { "R1 chat: reasoning_format \"none\"", "chat", R"({"reasoning_format": "none"})", 200, def_budget, nullptr,
      COMMON_REASONING_FORMAT_NONE, false },
    { "R1 completion: reasoning_format \"bogus\"", "completion", R"({"reasoning_format": "bogus"})", 400, any,
      "\"deepseek-legacy\"", -1, false },
    { "R1 completion: reasoning_format 5", "completion", R"({"reasoning_format": 5})", 400, any,
      "must be a string", -1, false },
    { "R1 completion: reasoning_format \"deepseek\"", "completion", R"({"reasoning_format": "deepseek"})", 200,
      def_budget, nullptr, COMMON_REASONING_FORMAT_DEEPSEEK, false },
    // R1: the budget's types, ranges, conflicts, inheritance and -1 through each route
    { "R1 chat: thinking_budget \"abc\"", "chat", R"({"thinking_budget": "abc"})", 400, any,
      "\"thinking_budget\" must be an integer", -1, false },
    { "R1 completion: thinking_budget_tokens 2.5", "completion", R"({"thinking_budget_tokens": 2.5})", 400, any,
      "\"thinking_budget_tokens\" must be an integer", -1, false },
    { "R1 chat: reasoning_budget_tokens -2", "chat", R"({"reasoning_budget_tokens": -2})", 400, any,
      "between -1 and", -1, false },
    { "R1 chat: aliases 10 and 11", "chat", R"({"reasoning_budget_tokens": 10, "thinking_budget": 11})", 400, any,
      "conflicting reasoning budgets", -1, false },
    { "R1 completion: aliases 10 and 11", "completion", R"({"reasoning_budget_tokens": 10, "thinking_budget": 11})",
      400, any, "conflicting reasoning budgets", -1, false },
    { "R1 chat: aliases 12 and 12", "chat", R"({"reasoning_budget": 12, "thinking_budget_tokens": 12})", 200, 12,
      nullptr, -1, false },
    { "R1 chat: no budget inherits the server's", "chat", "{}", 200, def_budget, nullptr, -1, false },
    { "R1 chat: null inherits the server's", "chat", R"({"reasoning_budget_tokens": null})", 200, def_budget,
      nullptr, -1, false },
    { "R1 chat: -1 is unlimited", "chat", R"({"reasoning_budget_tokens": -1})", 200, -1, nullptr, -1, false },
    { "R1 completion: no budget inherits the server's", "completion", "{}", 200, def_budget, nullptr, -1, false },
    { "R1 completion: reasoning_budget 9", "completion", R"({"reasoning_budget": 9})", 200, 9, nullptr, -1, false },
    { "R1 completion: thinking_budget -1", "completion", R"({"thinking_budget": -1})", 200, -1, nullptr, -1, false },
    { "R1 anthropic: no thinking inherits the server's", "anthropic", "{}", 200, def_budget, nullptr, -1, false },
    { "R1 anthropic: thinking enabled, budget_tokens 32", "anthropic",
      R"({"thinking": {"type": "enabled", "budget_tokens": 32}})", 200, 32, nullptr, -1, false },
    { "R1 anthropic: thinking enabled without budget_tokens (10000)", "anthropic",
      R"({"thinking": {"type": "enabled"}})", 200, 10000, nullptr, -1, false },
    { "R1 anthropic: thinking disabled inherits the server's", "anthropic", R"({"thinking": {"type": "disabled"}})",
      200, def_budget, nullptr, -1, false },
    { "R1 anthropic: budget_tokens -1", "anthropic", R"({"thinking": {"type": "enabled", "budget_tokens": -1}})", 200,
      -1, nullptr, -1, false },
    { "R1 anthropic: budget_tokens \"abc\"", "anthropic",
      R"({"thinking": {"type": "enabled", "budget_tokens": "abc"}})", 400, any,
      "\"thinking.budget_tokens\" must be an integer", -1, false },
    { "R1 anthropic: budget_tokens 3.5", "anthropic", R"({"thinking": {"type": "enabled", "budget_tokens": 3.5}})",
      400, any, "\"thinking.budget_tokens\" must be an integer", -1, false },
    { "R1 anthropic: budget_tokens -2", "anthropic", R"({"thinking": {"type": "enabled", "budget_tokens": -2}})", 400,
      any, "\"thinking.budget_tokens\" must be between -1 and", -1, false },
    // R2: misleading and misplaced budget controls
    { "R2 chat: max_reasoning_tokens 32", "chat", R"({"max_reasoning_tokens": 32})", 400, any,
      "\"reasoning_budget_tokens\"", -1, false },
    { "R2 completion: max_reasoning_tokens 32", "completion", R"({"max_reasoning_tokens": 32})", 400, any,
      "\"reasoning_budget_tokens\"", -1, false },
    { "R2 chat: chat_template_kwargs.reasoning_budget_tokens 32", "chat",
      R"({"chat_template_kwargs": {"reasoning_budget_tokens": 32}})", 400, any, "top-level \"reasoning_budget_tokens\"",
      -1, false },
    { "R2 chat: chat_template_kwargs.thinking_budget_tokens 32", "chat",
      R"({"chat_template_kwargs": {"thinking_budget_tokens": 32}})", 400, any, "top-level \"reasoning_budget_tokens\"",
      -1, false },
    { "R2 chat: chat_template_kwargs.reasoning_budget 32", "chat",
      R"({"chat_template_kwargs": {"reasoning_budget": 32}})", 400, any, "top-level \"reasoning_budget_tokens\"", -1,
      false },
    { "R2 chat: chat_template_kwargs.thinking_budget 32", "chat",
      R"({"chat_template_kwargs": {"thinking_budget": 32}})", 400, any, "top-level \"reasoning_budget_tokens\"", -1,
      false },
    { "R2 chat: chat_template_kwargs.max_reasoning_tokens 32", "chat",
      R"({"chat_template_kwargs": {"max_reasoning_tokens": 32}})", 400, any, "top-level \"reasoning_budget_tokens\"",
      -1, false },
    { "R2 completion: chat_template_kwargs.thinking_budget 32", "completion",
      R"({"chat_template_kwargs": {"thinking_budget": 32}})", 400, any, "top-level \"reasoning_budget_tokens\"", -1,
      false },
    { "R2 anthropic: chat_template_kwargs.thinking_budget 32", "anthropic",
      R"({"chat_template_kwargs": {"thinking_budget": 32}})", 400, any, "\"thinking.budget_tokens\"", -1, false },
    { "R2 chat: an Anthropic thinking.budget_tokens 32", "chat",
      R"({"thinking": {"type": "enabled", "budget_tokens": 32}})", 400, any, "/v1/messages", -1, false },
    { "R2 completion: an Anthropic thinking.budget_tokens 32", "completion",
      R"({"thinking": {"type": "enabled", "budget_tokens": 32}})", 400, any, "/v1/messages", -1, false },
    { "R2 anthropic: top-level reasoning_budget_tokens 32", "anthropic", R"({"reasoning_budget_tokens": 32})", 400,
      any, "\"thinking.budget_tokens\"", -1, false },
    { "R2 anthropic: top-level max_reasoning_tokens 32", "anthropic", R"({"max_reasoning_tokens": 32})", 400, any,
      "\"thinking.budget_tokens\"", -1, false },
    // R2: null is absent; the working fields and other template kwargs unchanged
    { "R2 chat: max_reasoning_tokens null", "chat", R"({"max_reasoning_tokens": null})", 200, def_budget, nullptr, -1,
      false },
    { "R2 chat: chat_template_kwargs.thinking_budget null", "chat",
      R"({"chat_template_kwargs": {"thinking_budget": null}})", 200, def_budget, nullptr, -1, false },
    { "R2 chat: thinking.budget_tokens null", "chat", R"({"thinking": {"type": "enabled", "budget_tokens": null}})",
      200, def_budget, nullptr, -1, false },
    { "R2 chat: chat_template_kwargs.enable_thinking false reaches the template", "chat",
      R"({"chat_template_kwargs": {"enable_thinking": false}})", 200, def_budget, nullptr, -1, true },
    { "R2 chat: thinking_budget 7 with chat_template_kwargs.enable_thinking true", "chat",
      R"({"thinking_budget": 7, "chat_template_kwargs": {"enable_thinking": true}})", 200, 7, nullptr, -1, false },
    // FBH1 A3: the OpenRouter object's reasoning.max_tokens is a budget alias on the OpenAI routes
    { "A3 chat: reasoning.max_tokens 48", "chat", R"({"reasoning": {"max_tokens": 48}})", 200, 48, nullptr, -1,
      false },
    { "A3 completion: reasoning.max_tokens 48", "completion", R"({"reasoning": {"max_tokens": 48}})", 200, 48,
      nullptr, -1, false },
    { "A3 chat: reasoning.max_tokens -1 is unlimited", "chat", R"({"reasoning": {"max_tokens": -1}})", 200, -1,
      nullptr, -1, false },
    { "A3 chat: reasoning.max_tokens 48 and thinking_budget 48", "chat",
      R"({"reasoning": {"max_tokens": 48}, "thinking_budget": 48})", 200, 48, nullptr, -1, false },
    { "A3 chat: reasoning.max_tokens 48 and reasoning_budget_tokens 49", "chat",
      R"({"reasoning": {"max_tokens": 48}, "reasoning_budget_tokens": 49})", 400, any,
      "conflicting reasoning budgets", -1, false },
    { "A3 completion: reasoning.max_tokens 48 and reasoning_budget 49", "completion",
      R"({"reasoning": {"max_tokens": 48}, "reasoning_budget": 49})", 400, any, "conflicting reasoning budgets", -1,
      false },
    { "A3 chat: reasoning.max_tokens \"x\"", "chat", R"({"reasoning": {"max_tokens": "x"}})", 400, any,
      "\"reasoning.max_tokens\" must be an integer", -1, false },
    { "A3 completion: reasoning.max_tokens -2", "completion", R"({"reasoning": {"max_tokens": -2}})", 400, any,
      "\"reasoning.max_tokens\" must be between -1 and", -1, false },
    { "A3 chat: reasoning 5", "chat", R"({"reasoning": 5})", 400, any, "\"reasoning\" must be an object", -1, false },
    { "A3 completion: reasoning \"high\"", "completion", R"({"reasoning": "high"})", 400, any,
      "\"reasoning\" must be an object", -1, false },
    { "A3 chat: reasoning without max_tokens inherits the server's", "chat", R"({"reasoning": {"effort": "high"}})",
      200, def_budget, nullptr, -1, false },
    { "A3 chat: reasoning.max_tokens null inherits the server's", "chat", R"({"reasoning": {"max_tokens": null}})",
      200, def_budget, nullptr, -1, false },
    { "A3 chat: reasoning null", "chat", R"({"reasoning": null})", 200, def_budget, nullptr, -1, false },
    { "A3 anthropic: reasoning.max_tokens 48", "anthropic", R"({"reasoning": {"max_tokens": 48}})", 400, any,
      "\"thinking.budget_tokens\"", -1, false },
    // FBH1 L5: a request that does not parse is an invalid request (400), never a server error
    { "L5 chat: a message's role is a number", "chat", R"({"messages": [{"role": 5, "content": "x"}]})", 400, any,
      "Failed to parse messages", -1, false },
    { "L5 chat: tools is a string", "chat", R"({"tools": "x"})", 400, any, "Failed to parse tools", -1, false },
    { "L5 anthropic: messages is a string", "anthropic", R"({"messages": "x"})", 400, any, nullptr, -1, false },
};

// FBH1 A2: the budget message through each route. message: the effective text (null: not checked; flag_text: the
// server's --reasoning-budget-message)
const char * const flag_text = "\n\nTime is up, answer now.";

struct mcase {
    const char * name;
    const char * route;
    const char * extra;
    int          status;
    const char * message;
    const char * has;
};

const mcase mcases[] = {
    { "A2 chat: no reasoning_budget_message inherits the flag's", "chat", "{}", 200, flag_text, nullptr },
    { "A2 chat: reasoning_budget_message null inherits the flag's", "chat", R"({"reasoning_budget_message": null})",
      200, flag_text, nullptr },
    { "A2 chat: reasoning_budget_message \"stop\"", "chat", R"({"reasoning_budget_message": "stop"})", 200, "stop",
      nullptr },
    { "A2 chat: reasoning_budget_message \"\" turns it off", "chat", R"({"reasoning_budget_message": ""})", 200, "",
      nullptr },
    { "A2 chat: reasoning_budget_message 5", "chat", R"({"reasoning_budget_message": 5})", 400, nullptr,
      "Field 'reasoning_budget_message': \"reasoning_budget_message\" must be a string" },
    { "A2 chat: reasoning_budget_message [\"a\"]", "chat", R"({"reasoning_budget_message": ["a"]})", 400, nullptr,
      "Field 'reasoning_budget_message': \"reasoning_budget_message\" must be a string" },
    { "A2 completion: no reasoning_budget_message inherits the flag's", "completion", "{}", 200, flag_text, nullptr },
    { "A2 completion: reasoning_budget_message \"wrap up\"", "completion",
      R"({"reasoning_budget_message": "\n\nwrap up"})", 200, "\n\nwrap up", nullptr },
    { "A2 completion: reasoning_budget_message \"\" turns it off", "completion", R"({"reasoning_budget_message": ""})",
      200, "", nullptr },
    { "A2 completion: reasoning_budget_message true", "completion", R"({"reasoning_budget_message": true})", 400,
      nullptr, "Field 'reasoning_budget_message': \"reasoning_budget_message\" must be a string" },
    { "A2 anthropic: no reasoning_budget_message inherits the flag's", "anthropic", "{}", 200, flag_text, nullptr },
    { "A2 anthropic: reasoning_budget_message \"wrap up\" with a budget", "anthropic",
      R"({"reasoning_budget_message": "wrap up", "thinking": {"type": "enabled", "budget_tokens": 32}})", 200,
      "wrap up", nullptr },
    { "A2 anthropic: reasoning_budget_message \"\" turns it off", "anthropic", R"({"reasoning_budget_message": ""})",
      200, "", nullptr },
    { "A2 anthropic: reasoning_budget_message 7", "anthropic", R"({"reasoning_budget_message": 7})", 400, nullptr,
      "Field 'reasoning_budget_message': \"reasoning_budget_message\" must be a string" },
    { "A2 anthropic: reasoning_budget_message {}", "anthropic", R"({"reasoning_budget_message": {}})", 400, nullptr,
      "Field 'reasoning_budget_message': \"reasoning_budget_message\" must be a string" },
};

} // namespace

int reasoning_check_protocol(const reasoning_check_ctx & c) {
    int                 fails       = 0;
    const int           saved       = c.opt.reasoning_budget;
    common_params       params      = c.params_base;
    params.sampling.reasoning_budget_tokens = def_budget;
    c.opt.reasoning_budget                  = def_budget;
    const json   user  = json::array({ { { "role", "user" }, { "content", "q" } } });
    const auto   base  = [&](const std::string & r) {
        if (r == "completion") {
            return json{ { "prompt", "q" } };
        }
        json b = { { "messages", user } };
        if (r == "anthropic") {
            b["max_tokens"] = 16;
        }
        return b;
    };
    const std::string plain = route(c, params, "chat", base("chat")).prompt;
    for (const pcase & k : cases) {
        json       body  = base(k.route);
        const json extra = json::parse(k.extra);
        for (const auto & it : extra.items()) {
            body[it.key()] = it.value();
        }
        const reply r  = route(c, params, k.route, body);
        const bool  ok = r.status == k.status && (k.budget == any || r.budget == k.budget) &&
                        (k.has == nullptr || r.error.find(k.has) != std::string::npos) &&
                        (k.format < 0 || r.format == k.format) && (!k.templated || r.prompt != plain);
        std::string got = "status " + std::to_string(r.status);
        got += r.budget == any ? std::string() : ", budget " + std::to_string(r.budget);
        got += r.error.empty() ? std::string() : ", error " + json(r.error).dump();
        got += k.templated ? std::string(r.prompt != plain ? ", prompt templated" : ", prompt unchanged") : "";
        std::printf("reasoning-check: %s %s %s: %s\n", ok ? "PASS" : "FAIL", k.route, k.extra, got.c_str());
        std::printf("reasoning-check:      (%s; want status %d%s%s)\n", k.name, k.status,
                    k.budget == any ? "" : (", budget " + std::to_string(k.budget)).c_str(),
                    k.has == nullptr ? "" : (", error with " + json(std::string(k.has)).dump()).c_str());
        fails += ok ? 0 : 1;
    }
    c.opt.reasoning_budget = saved;
    std::printf("reasoning-check: protocol %s (%d of %zu failed)\n", fails == 0 ? "PASS" : "FAIL", fails,
                sizeof(cases) / sizeof(cases[0]));

    // FBH1 A2: the budget message: the flag's default, then the field on each route with the flag set to flag_text
    int        mfails = 0;
    size_t     mtotal = 0;
    const auto mark   = [&](bool ok, const std::string & line) {
        std::printf("reasoning-check: %s %s\n", ok ? "PASS" : "FAIL", line.c_str());
        mfails += ok ? 0 : 1;
        ++mtotal;
    };
    const std::string def_message = "\n\nConsidering the limited time by the user, I have to give the final answer "
                                    "directly now, briefly, without any further work.";
    mark(common_params().sampling.reasoning_budget_message == def_message,
         "A2 flag: --reasoning-budget-message defaults to a blank line and the early-stop sentence");
    {
        // the check's server default (untouched unless the check was started with the flag)
        const reply r = route(c, params, "chat", base("chat"));
        mark(r.status == 200 && r.message == c.params_base.sampling.reasoning_budget_message,
             "A2 chat: no field and no flag change: the server's message, " + json(r.message).dump());
    }
    params.sampling.reasoning_budget_message = flag_text;
    for (const mcase & k : mcases) {
        json       body  = base(k.route);
        const json extra = json::parse(k.extra);
        for (const auto & it : extra.items()) {
            body[it.key()] = it.value();
        }
        const reply r  = route(c, params, k.route, body);
        const bool  ok = r.status == k.status && (k.message == nullptr || r.message == k.message) &&
                        (k.has == nullptr || r.error.find(k.has) != std::string::npos);
        std::string got = "status " + std::to_string(r.status);
        got += r.status == 200 ? ", message " + json(r.message).dump() : ", error " + json(r.error).dump();
        mark(ok, std::string(k.route) + " " + k.extra + ": " + got);
        std::printf("reasoning-check:      (%s; want status %d%s%s)\n", k.name, k.status,
                    k.message == nullptr ? "" : (", message " + json(std::string(k.message)).dump()).c_str(),
                    k.has == nullptr ? "" : (", error with " + json(std::string(k.has)).dump()).c_str());
    }
    {
        // a message over the engine's limit is the client's error, on each route
        std::string text;
        for (int i = 0; i < 400; ++i) {
            text += " word" + std::to_string(i);
        }
        for (const char * r : { "chat", "completion", "anthropic" }) {
            json body                        = base(r);
            body["reasoning_budget_message"] = text;
            const reply x                    = route(c, params, r, body);
            mark(x.status == 400 && x.error.find("Field 'reasoning_budget_message'") != std::string::npos &&
                     x.error.find("more than 256") != std::string::npos,
                 std::string(r) + " a message of 400 words: status " + std::to_string(x.status) + ", error " +
                     json(x.error).dump());
        }
    }
    {
        // the tokens: plain text, no control tokens parsed; a tag's text is refused where the tags are given (the
        // server's admission check: Qwen's </think> is a user-defined token, so its text would be the tag's own token)
        std::vector<int32_t> ids;
        std::string          err;
        const bool           taken = think_message_ids(c.vocab, def_message, { "</think>", "<think>" }, ids, err);
        const std::string    back  = common_detokenize(c.vocab, llama_tokens(ids.begin(), ids.end()), false);
        mark(taken && !ids.empty() && ids.size() <= (size_t) think_message_max && back == def_message,
             "A2 tokens: the default message is " + std::to_string(ids.size()) + " plain-text tokens that read back "
             "as the text");
        const bool tagged = think_message_ids(c.vocab, "stop </think> now", { "</think>", "<think>" }, ids, err);
        mark(!tagged && ids.empty() && err.find("</think>") != std::string::npos,
             "A2 tokens: a message holding a reasoning tag's text is refused where the tags are given (" + err + ")");
        mark(think_message_ids(c.vocab, "", { "</think>" }, ids, err) && ids.empty(),
             "A2 tokens: an empty message has no tokens");
    }
    {
        // FBH1 A3: usage reports the reasoning tokens with and without a budget
        server_task_result_cmpl_final f;
        f.n_decoded             = 40;
        f.n_prompt_tokens       = 10;
        f.n_prompt_tokens_cache = 0;
        f.stats.think_budget    = -1;
        f.stats.n_think         = 7;
        const json u0           = f.usage_json_oaicompat();
        f.stats.think_budget    = 64;
        f.stats.n_think         = 64;
        f.stats.n_think_extra   = 2;
        const json u1           = f.usage_json_oaicompat();
        f.stats                 = {};
        const json u2           = f.usage_json_oaicompat();
        const auto n            = [](const json & u) {
            return u.contains("completion_tokens_details")
                       ? json_value(u.at("completion_tokens_details"), "reasoning_tokens", (int64_t) -1)
                       : (int64_t) -1;
        };
        mark(n(u0) == 7 && n(u1) == 66 && n(u2) == 0,
             "A3 usage: completion_tokens_details.reasoning_tokens without a budget " + std::to_string(n(u0)) +
                 " (7 counted), with one " + std::to_string(n(u1)) + " (64 counted + 2 finishing a character), none "
                 "counted " + std::to_string(n(u2)));
    }
    {
        // FBH1 L5: a body that is not JSON, as a route's json::parse throws it to ex_wrapper
        int code = 0;
        try {
            (void) json::parse(std::string("{\"messages\": ["));
        } catch (...) {
            code = json_value(format_exception_response(std::current_exception()), "code", 0);
        }
        mark(code == 400, "L5 a body that is not JSON: status " + std::to_string(code) + " (400)");
    }
    std::printf("reasoning-check: message, usage and bad body %s (%d of %zu failed)\n",
                mfails == 0 ? "PASS" : "FAIL", mfails, mtotal);
    return fails + mfails;
}

//
// R3: UTF-8 at the budget
//

namespace {

// the check's own UTF-8 decoder, a byte at a time (need: the continuation bytes the character lacks); false at a byte
// that cannot come there
bool dec(int32_t & need, unsigned char b) {
    if ((b & 0xC0) == 0x80) {
        if (need > 0) {
            --need;
            return true;
        }
        return false;
    }
    const bool ok = need == 0;
    need = b < 0x80 ? 0 : b >= 0xC2 && b <= 0xDF ? 1 : b >= 0xE0 && b <= 0xEF ? 2 : b >= 0xF0 && b <= 0xF4 ? 3 : -1;
    if (need < 0) {
        need = 0;
        return false;
    }
    return ok;
}

bool dec_all(int32_t & need, const std::string & s) {
    bool ok = true;
    for (const char ch : s) {
        ok = dec(need, static_cast<unsigned char>(ch)) && ok;
    }
    return ok;
}

bool whole(const std::string & s) {
    int32_t need = 0;
    return dec_all(need, s) && need == 0;
}

// FBH1 L1: the check's own reading of a piece that ends a sentence: a newline last, or, trailing spaces aside, one
// of . ! ? or their CJK forms (U+3002, U+FF01, U+FF1F)
bool ends_sentence(const std::string & s) {
    if (!s.empty() && s.back() == '\n') {
        return true;
    }
    const size_t n = s.find_last_not_of(' ');
    if (n == std::string::npos) {
        return false;
    }
    const std::string t = s.substr(0, n + 1);
    for (const char * e : { ".", "!", "?", "\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F" }) {
        const std::string x = e;
        if (t.size() >= x.size() && t.compare(t.size() - x.size(), x.size(), x) == 0) {
            return true;
        }
    }
    return false;
}

struct ucase {
    std::string          name;
    std::vector<int32_t> reason, answer;
    int32_t              budget = 0, limit = 400;
    bool                 split  = false; // the budget lands inside a character (the old tracker breaks it)
};

struct outcome {
    std::vector<int32_t> g;
    int32_t              counted = 0, forced = 0, extra = 0;
    bool                 reasoning = true;
};

// the rule from the bytes (utf8 false: the old tracker): the listed model's reasoning, its own close when it runs out,
// the answer and eos; one-token sequences (each engine's model)
outcome simulate(const ucase & k, const std::vector<int32_t> & close, const std::vector<std::vector<int32_t>> & ends,
                 int32_t eos, const std::function<std::string(int32_t)> & piece, bool utf8) {
    outcome o;
    const auto ending = [&](int32_t t) {
        bool e = t == close.back();
        for (const std::vector<int32_t> & x : ends) {
            e = e || x.back() == t;
        }
        return e;
    };
    size_t  r = 0, a = 0;
    int32_t need = 0, waited = 0;
    bool    broken = false;
    while (static_cast<int32_t>(o.g.size()) < k.limit && (o.g.empty() || o.g.back() != eos)) {
        if (!o.reasoning) {
            o.g.push_back(a < k.answer.size() ? k.answer[a++] : eos);
            continue;
        }
        const bool spent = o.counted >= k.budget;
        if (spent && !(utf8 && need > 0 && waited < 3 && !broken)) {
            o.g.push_back(close[0]);
            ++o.forced;
            o.reasoning = false;
            continue;
        }
        const int32_t t = r < k.reason.size() ? k.reason[r] : close[0];
        o.g.push_back(t);
        if (ending(t)) {
            o.reasoning = false;
            continue;
        }
        ++r;
        const bool ok = dec_all(need, piece(t));
        if (spent) {
            ++waited;
            ++o.extra;
            broken = broken || !ok;
        } else {
            ++o.counted;
        }
    }
    return o;
}

json msg_json(const common_chat_msg & m) {
    return { { "reasoning_content", m.reasoning_content }, { "content", m.content } };
}

void put_list(FILE * f, const char * key, const std::vector<int32_t> & v) {
    std::fprintf(f, "%s %zu", key, v.size());
    for (const int32_t x : v) {
        std::fprintf(f, " %d", x);
    }
    std::fprintf(f, "\n");
}

} // namespace

int reasoning_check_utf8(const reasoning_check_ctx & c, const std::string & model, const std::vector<int32_t> & open,
                         const std::vector<int32_t> & close, const std::vector<std::vector<int32_t>> & ends,
                         uint32_t (*entry)(int32_t, const char *, size_t), const std::string & fixture) {
    int        fails  = 0;
    const auto report = [&](bool ok, const std::string & what) {
        std::printf("reasoning-check: %s utf8 %s\n", ok ? "PASS" : "FAIL", what.c_str());
        fails += ok ? 0 : 1;
    };
    if (close.size() != 1) {
        report(false, "the closing sequence is not one token");
        return fails;
    }
    // the table as the server builds it at load, and every token's piece as the table read it
    const bool special = c.params_base.special;
    const auto table   = think_utf8_table(c.vocab, special, entry);
    const int32_t nv   = llama_vocab_n_tokens(c.vocab);
    std::vector<std::string> pieces(static_cast<size_t>(nv));
    std::vector<int32_t>     byte_tok(256, -1);
    for (int32_t id = 0; id < nv; ++id) {
        pieces[static_cast<size_t>(id)] = common_token_to_piece(c.vocab, id, special);
        const std::string & p = pieces[static_cast<size_t>(id)];
        if (p.size() == 1 && byte_tok[static_cast<unsigned char>(p[0])] < 0) {
            byte_tok[static_cast<unsigned char>(p[0])] = id;
        }
    }
    {
        size_t at = 0, wrong = 0, first = SIZE_MAX;
        for (int32_t id = 0; id < nv; ++id) {
            const bool listed = at < table->size() && static_cast<int32_t>((*table)[at] & 0xFFFFFF) == id;
            at += listed ? 1 : 0;
            const std::string & p = pieces[static_cast<size_t>(id)];
            if (listed != (p.empty() || !whole(p) || ends_sentence(p))) {
                ++wrong;
                first = std::min(first, static_cast<size_t>(id));
            }
        }
        report(wrong == 0 && at == table->size(),
               "the server's table lists " + std::to_string(table->size()) + " of " + std::to_string(nv) +
                   " tokens: exactly those whose piece is empty, not whole UTF-8 characters by the check's own "
                   "decoder, or ends a sentence (FBH1 L1)" +
                   (wrong ? " (" + std::to_string(wrong) + " wrong, the first " + std::to_string(first) + ")" : ""));
    }
    const auto piece = [&](int32_t t) { return t >= 0 && t < nv ? pieces[static_cast<size_t>(t)] : std::string(); };
    const auto tok   = [&](const std::string & s) {
        std::vector<int32_t> v;
        for (const llama_token t : common_tokenize(c.vocab, s, false, false)) {
            v.push_back(t);
        }
        return v;
    };
    const auto cat = [](std::vector<int32_t> a, const std::vector<int32_t> & b) {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    };
    const auto bytes_of = [&](const std::string & s) {
        std::vector<int32_t> v;
        for (const char ch : s) {
            v.push_back(byte_tok[static_cast<unsigned char>(ch)]);
        }
        return v;
    };
    bool have_bytes = true;
    for (const char * ch : { "\xC3\xA9", "\xE4\xB8\xAD", "\xF0\x9F\x98\x80", "a" }) {
        for (const int32_t t : bytes_of(ch)) {
            have_bytes = have_bytes && t >= 0;
        }
    }
    report(have_bytes, "the vocabulary has a token for each byte the constructed cases use");
    if (!have_bytes) {
        return fails;
    }
    const std::vector<int32_t> pre    = tok("We need to count them. ");
    const std::vector<int32_t> tail   = tok(" Then answer.");
    const std::vector<int32_t> answer = tok("There are 17.");
    const auto                 np     = static_cast<int32_t>(pre.size());
    std::vector<ucase>         cases;
    const auto add = [&](const std::string & name, std::vector<int32_t> reason, int32_t budget, bool split,
                         int32_t limit = 400) {
        ucase k;
        k.name   = name;
        k.reason = std::move(reason);
        k.answer = answer;
        k.budget = budget;
        k.split  = split;
        k.limit  = limit;
        cases.push_back(k);
    };
    add("ascii", cat(pre, tail), np, false);
    // the tokenizer's own splits: a rare character of each length inside a sentence, the budget after the token that
    // ends inside it
    const std::vector<std::vector<const char *>> rare = {
        { "\xCA\xAC", "\xD3\x82", "\xD4\x93", "\xDD\xA9", "\xDE\x97", "\xC9\xAE", "\xCF\x97" },
        { "\xE9\xBE\x98", "\xE4\xA8\xBB", "\xE9\xB1\xBB", "\xEA\x99\xAE", "\xE2\xB6\x86", "\xEA\x95\xA2" },
        { "\xF0\xA0\x9C\x8E", "\xF0\x9D\x94\x98", "\xF0\x9F\xAB\xA0", "\xF0\x93\x80\x80", "\xF0\x9F\xA6\xA9",
          "\xF0\xA0\x80\x8B" },
    };
    for (size_t len = 2; len <= 4; ++len) {
        bool found = false;
        for (const char * ch : rare[len - 2]) {
            const std::string          text = "We need " + std::string(ch) + " now.";
            const std::vector<int32_t> v    = tok(text);
            const size_t               s0   = 8; // the character's first byte
            size_t                     end  = 0;
            for (size_t i = 0; i < v.size() && !found; ++i) {
                end += piece(v[i]).size();
                if (end > s0 && end < s0 + len) {
                    add("tokenizer-" + std::to_string(len) + "-byte", v, static_cast<int32_t>(i) + 1, true);
                    found = true;
                }
            }
            if (found) {
                std::string hex;
                for (const char b : std::string(ch)) {
                    char h[4];
                    std::snprintf(h, sizeof h, "%02X", static_cast<unsigned char>(b));
                    hex += h;
                }
                std::printf("reasoning-check: utf8 the tokenizer splits U+%s bytes %s over %zu tokens of \"We need "
                            "... now.\"\n", ch, hex.c_str(), v.size());
                break;
            }
        }
        if (!found) {
            std::printf("reasoning-check: utf8 no %zu-byte character of the list splits in this tokenizer (the byte "
                        "cases below cover it)\n", len);
        }
    }
    add("bytes-2-after-lead", cat(cat(pre, bytes_of("\xC3\xA9")), tail), np + 1, true);
    add("bytes-3-after-lead", cat(cat(pre, bytes_of("\xE4\xB8\xAD")), tail), np + 1, true);
    add("bytes-3-after-second", cat(cat(pre, bytes_of("\xE4\xB8\xAD")), tail), np + 2, true);
    add("bytes-4-after-lead", cat(cat(pre, bytes_of("\xF0\x9F\x98\x80")), tail), np + 1, true);
    add("bytes-4-after-third", cat(cat(pre, bytes_of("\xF0\x9F\x98\x80")), tail), np + 3, true);
    add("bytes-on-a-boundary", cat(cat(pre, bytes_of("\xC3\xA9")), tail), np + 2, false);
    add("broken-by-ascii", cat(cat(pre, bytes_of("\xF0" "a")), tail), np + 1, true);
    add("the-models-close-while-waiting", cat(pre, bytes_of("\xE4")), np + 1, true);
    add("total-limit-while-waiting", cat(cat(pre, bytes_of("\xF0\x9F\x98\x80")), tail), np + 1, true, np + 3);
    // a token that finishes one character and begins another, if the vocabulary has one: the lead byte it continues,
    // then it, then its last character's missing bytes
    for (int32_t id = 0; id < nv; ++id) {
        const std::string & p = pieces[static_cast<size_t>(id)];
        size_t              l = 0;
        while (l < p.size() && (static_cast<unsigned char>(p[l]) & 0xC0) == 0x80) {
            ++l;
        }
        int32_t need = 0;
        // its character done within the wait: 1 + need extra tokens, at most 3
        if (l == 0 || l > 3 || l == p.size() || !dec_all(need, p.substr(l)) || need == 0 || need > 2) {
            continue;
        }
        const char *         leads[] = { "", "\xC3", "\xE4", "\xF0" };
        std::vector<int32_t> r       = cat(pre, bytes_of(leads[l]));
        r.push_back(id);
        r = cat(cat(r, bytes_of(std::string(static_cast<size_t>(need), '\x80'))), tail);
        add("a-token-finishing-one-character-and-beginning-another-" + std::to_string(id), r, np + 1, true);
        break;
    }

    // the request's parser and the shell's rendering (server-context.cpp process_token: special or preserved)
    json body = { { "messages", json::array({ { { "role", "user" }, { "content", "q" } } }) },
                  { "chat_template_kwargs", { { "enable_thinking", true } } } };
    const json        data = oaicompat_chat_params_parse(body, c.opt);
    const task_params tp   = server_schema::eval_llama_cmpl_schema(c.vocab, c.params_base, c.logit_bias_eog, data);
    const common_chat_parser_params & pp  = tp.chat_parser_params;
    const auto out_piece = [&](int32_t t) {
        return common_token_to_piece(c.vocab, t, special || tp.sampling.preserved_tokens.count(t) > 0);
    };
    struct parsed {
        json        fin, str;
        bool        held = false; // the text ends inside a character: the stream holds those bytes
        std::string fin_dump;
    };
    const auto parse = [&](const std::vector<int32_t> & g) {
        parsed                            r;
        std::string                       gen;
        size_t                            sent = 0;
        task_result_state                 st(pp);
        std::vector<common_chat_msg_diff> diffs;
        common_chat_msg                   streamed;
        const auto take = [&]() {
            for (const common_chat_msg_diff & d : diffs) {
                streamed.content += d.content_delta;
                streamed.reasoning_content += d.reasoning_content_delta;
            }
            diffs.clear();
        };
        for (const int32_t t : g) {
            gen += out_piece(t);
            if (validate_utf8(gen) == gen.size()) {
                st.update_chat_msg(gen.substr(sent), true, diffs);
                sent = gen.size();
                take();
            }
        }
        st.update_chat_msg("", false, diffs);
        take();
        r.held     = sent < gen.size();
        // the final reply as the shell parses it (server_task_result_cmpl_final::update: the whole text, not partial)
        task_result_state                 fs(pp);
        std::vector<common_chat_msg_diff> fd;
        r.fin      = msg_json(fs.update_chat_msg(gen, false, fd));
        r.str      = msg_json(streamed);
        r.fin_dump = safe_json_to_str(r.fin);
        return r;
    };
    const std::string fffd = "\xEF\xBF\xBD";
    std::string       answer_text;
    for (const int32_t t : answer) {
        answer_text += out_piece(t);
    }
    const int32_t     eos  = llama_vocab_eos(c.vocab) != LLAMA_TOKEN_NULL ? llama_vocab_eos(c.vocab) :
                                                                            llama_vocab_eot(c.vocab);

    FILE * f = std::fopen(fixture.c_str(), "w");
    if (f == nullptr) {
        report(false, "cannot write the fixture " + fixture);
        return fails;
    }
    std::fprintf(f, "think-utf8 %s\n", model.c_str());
    put_list(f, "open", open);
    put_list(f, "close", close);
    std::fprintf(f, "ends %zu\n", ends.size());
    for (const std::vector<int32_t> & e : ends) {
        put_list(f, "", e);
    }
    std::fprintf(f, "eos %d\n", eos);
    std::vector<int32_t> t32;
    for (const uint32_t e : *table) {
        t32.push_back(static_cast<int32_t>(e)); // the entry's bits (negative with the sentence-end class bit)
    }
    put_list(f, "table", t32);
    std::fprintf(f, "cases %zu\n", cases.size());
    for (const ucase & k : cases) {
        const outcome h = simulate(k, close, ends, eos, piece, true);
        const outcome b = simulate(k, close, ends, eos, piece, false);
        std::fprintf(f, "case %s %d %d\n", k.name.c_str(), k.budget, k.limit);
        put_list(f, "reason", k.reason);
        put_list(f, "answer", k.answer);
        for (const auto & [key, o] : { std::make_pair("want", h), std::make_pair("base", b) }) {
            std::fprintf(f, "%s %zu", key, o.g.size());
            for (const int32_t x : o.g) {
                std::fprintf(f, " %d", x);
            }
            std::fprintf(f, " %d %d %d %d\n", o.counted, o.forced, o.extra, o.reasoning ? 1 : 0);
        }
        parsed      ph, pb;
        std::string threw;
        try {
            ph = parse(h.g);
            pb = parse(b.g);
        } catch (const std::exception & e) {
            threw = e.what();
        }
        if (!threw.empty()) {
            report(false, k.name + ": the parse threw: " + threw);
            continue;
        }
        // stream and final agree; a text ending inside a character keeps those bytes from the stream (the shell's
        // output buffer), so its streamed reasoning is the final one less them
        const auto agree = [](const parsed & p) {
            if (!p.held) {
                return p.fin == p.str;
            }
            const std::string fr = p.fin.at("reasoning_content").get<std::string>();
            const std::string sr = p.str.at("reasoning_content").get<std::string>();
            return fr.compare(0, sr.size(), sr) == 0 && validate_utf8(fr) == sr.size();
        };
        // the reply: valid UTF-8 always; U+FFFD only where the model's own bytes are broken (the rule cannot mend
        // them); after a close the answer is the content and the reasoning stays reasoning
        const bool h_whole  = whole(ph.fin.at("reasoning_content").get<std::string>()) &&
                              whole(ph.fin.at("content").get<std::string>());
        const bool h_fffd   = ph.fin_dump.find(fffd) != std::string::npos;
        const bool broken   = k.name == "broken-by-ascii" || k.name == "the-models-close-while-waiting";
        const bool answered = ph.fin.at("content").get<std::string>() == answer_text;
        bool       ok       = agree(ph) && h_whole && h_fffd == broken && (h.reasoning || answered);
        // the old tracker closes inside the character: its reply shows the break (U+FFFD, or the reasoning in the
        // content)
        const bool b_bad = pb.fin_dump.find(fffd) != std::string::npos ||
                           pb.fin.at("content").get<std::string>() != answer_text;
        if (k.split && h.forced > 0) {
            ok = ok && b_bad;
        }
        if (k.name.rfind("bytes-", 0) == 0 || k.name.rfind("tokenizer-", 0) == 0 || k.name == "ascii" ||
            k.name.rfind("a-token-", 0) == 0) {
            ok = ok && h.forced == 1;
        }
        report(ok, k.name + ": budget " + std::to_string(k.budget) + ", counted " + std::to_string(h.counted) +
                       ", extra " + std::to_string(h.extra) + ", forced " + std::to_string(h.forced) +
                       (h.reasoning ? ", unclosed" : "") + (h_fffd ? "; U+FFFD for the model's broken bytes" : "") +
                       ", final " + ph.fin_dump +
                       (agree(ph) ? ", the stream the same" : ", the stream DIFFERS " + safe_json_to_str(ph.str)) +
                       (ph.held ? " (its incomplete bytes held)" : "") +
                       " | old tracker: extra " + std::to_string(b.extra) +
                       ", final " + pb.fin_dump + (agree(pb) ? "" : ", streamed " + safe_json_to_str(pb.str)));
    }
    std::fclose(f);
    std::printf("reasoning-check: utf8 %s (%d failed), %zu cases written to %s\n", fails == 0 ? "PASS" : "FAIL", fails,
                cases.size(), fixture.c_str());
    return fails;
}

//
// FBH1 K2: the model's own closing tag after a forced close
//

namespace {

struct shown {
    std::string content, reasoning;
};

// the texts a route's reply JSON shows: the chat route's message or deltas, the Anthropic route's blocks or events
void collect(const json & j, shown & s) {
    if (j.is_array()) {
        for (const json & x : j) {
            collect(x, s);
        }
        return;
    }
    if (!j.is_object()) {
        return;
    }
    const auto str = [&](const json & o, const char * key) {
        return o.contains(key) && o.at(key).is_string() ? o.at(key).get<std::string>() : std::string();
    };
    if (j.contains("choices")) { // the chat route: a whole message, or a chunk's delta
        for (const json & ch : j.at("choices")) {
            const json & m = ch.contains("message") ? ch.at("message") : ch.at("delta");
            s.content += str(m, "content");
            s.reasoning += str(m, "reasoning_content");
        }
        return;
    }
    if (j.contains("event")) { // an Anthropic stream event
        const json & d = j.at("data");
        if (d.contains("delta")) {
            s.content += str(d.at("delta"), "text");
            s.reasoning += str(d.at("delta"), "thinking");
        }
        return;
    }
    if (j.contains("content") && j.at("content").is_array()) { // an Anthropic message
        for (const json & b : j.at("content")) {
            s.content += str(b, "text");
            s.reasoning += str(b, "thinking");
        }
    }
}

struct ktok {
    int32_t tok;
    bool    forced;
};

} // namespace

int reasoning_check_close(const reasoning_check_ctx & c, const std::vector<int32_t> & open,
                          const std::vector<int32_t> & close, const std::vector<std::vector<int32_t>> & ends) {
    int        fails  = 0;
    const auto report = [&](bool ok, const std::string & what) {
        std::printf("reasoning-check: %s close %s\n", ok ? "PASS" : "FAIL", what.c_str());
        fails += ok ? 0 : 1;
    };
    if (close.size() != 1) {
        report(false, "the closing sequence is not one token");
        return fails;
    }
    // K1's form as the server tokenizes it at load
    std::vector<int32_t> lead, tail;
    think_form_ids(c.vocab, open, close, ends, lead, tail);
    const auto ids = [](const std::vector<int32_t> & v) {
        std::string s;
        for (const int32_t x : v) {
            s += (s.empty() ? "" : " ") + std::to_string(x);
        }
        return s;
    };
    report(!lead.empty() && !tail.empty(),
           "the forced close's form: a newline [" + ids(lead) + "] before the closing sequence [" + ids(close) +
               "] and a blank line [" + ids(tail) + "] after it");
    if (lead.empty() || tail.empty()) {
        return fails;
    }
    json body = { { "messages", json::array({ { { "role", "user" }, { "content", "q" } } }) },
                  { "chat_template_kwargs", { { "enable_thinking", true } } } };
    const json        data = oaicompat_chat_params_parse(body, c.opt);
    const task_params tp   = server_schema::eval_llama_cmpl_schema(c.vocab, c.params_base, c.logit_bias_eog, data);
    const auto out_piece = [&](int32_t t) {
        return common_token_to_piece(c.vocab, t, c.params_base.special || tp.sampling.preserved_tokens.count(t) > 0);
    };
    const auto text = [&](const std::vector<ktok> & g) {
        std::string s;
        for (const ktok & k : g) {
            s += out_piece(k.tok);
        }
        return s;
    };
    const auto seq = [&](std::vector<ktok> & g, const std::string & s, bool forced = false) {
        for (const llama_token t : common_tokenize(c.vocab, s, false, false)) {
            g.push_back({ t, forced });
        }
    };
    const auto put = [](std::vector<ktok> & g, const std::vector<int32_t> & v, bool forced) {
        for (const int32_t t : v) {
            g.push_back({ t, forced });
        }
    };
    // a reply through the filter and one route's result classes, as apply_step, process_token and the two send
    // functions fill them; chat_route false: the filter as the raw completion routes set it (off)
    const auto run = [&](const std::vector<ktok> & g, task_response_type rt, bool stream, bool chat_route) {
        think_close_hide hide;
        hide.reset(close, chat_route);
        task_result_state st(tp.chat_parser_params);
        json              out = json::array();
        std::string       gen;
        int32_t           n = 0;
        for (const ktok & k : g) {
            std::string piece = out_piece(k.tok);
            if (hide.hide(k.tok, k.forced, piece)) {
                piece.clear();
            }
            gen += piece;
            ++n;
            if (stream) {
                server_task_result_cmpl_partial p;
                p.content               = piece;
                p.n_decoded             = n;
                p.n_prompt_tokens       = 1;
                p.n_prompt_tokens_cache = 0;
                p.post_sampling_probs   = false;
                p.res_type              = rt;
                p.update(st);
                out.push_back(p.to_json());
            }
        }
        server_task_result_cmpl_final f;
        f.content               = stream ? std::string() : gen;
        f.stream                = stream;
        f.include_usage         = false;
        f.truncated             = false;
        f.n_decoded             = n;
        f.n_prompt_tokens       = 1;
        f.n_prompt_tokens_cache = 0;
        f.n_tokens_cached       = 0;
        f.has_new_line          = false;
        f.stop                  = STOP_TYPE_EOS;
        f.post_sampling_probs   = false;
        f.res_type              = rt;
        f.update(st);
        out.push_back(f.to_json());
        shown s;
        collect(out, s);
        return s;
    };
    const std::string think  = "We need to count the primes below 60";
    const std::string answer = "There are 17.";
    const std::string tag    = out_piece(close[0]);
    // the forced close: the reasoning, then the newline, the closing token and the blank line, each forced
    const auto forced_close = [&]() {
        std::vector<ktok> g;
        seq(g, think);
        put(g, lead, true);
        put(g, close, true);
        put(g, tail, true);
        return g;
    };
    struct kcase {
        std::string       name;
        std::vector<ktok> g;
        std::string       want; // the content with the filter; empty: the same as without it
        bool              fault = false; // the filter's case: without it the content begins with the closing tag
    };
    std::vector<kcase> cases;
    {
        kcase k{ "the model's own closing tag and blank line after the forced close", forced_close(), answer, true };
        put(k.g, close, false);
        seq(k.g, "\n\n" + answer);
        cases.push_back(k);
    }
    {
        kcase k{ "its newline, closing tag and blank line after the forced close", forced_close(), answer, true };
        seq(k.g, "\n");
        put(k.g, close, false);
        seq(k.g, "\n\n" + answer);
        cases.push_back(k);
    }
    {
        kcase k{ "two closing tags of its own after the forced close", forced_close(), answer, true };
        put(k.g, close, false);
        seq(k.g, "\n\n");
        put(k.g, close, false);
        seq(k.g, "\n\n" + answer);
        cases.push_back(k);
    }
    {
        kcase k{ "the answer directly after the forced close (nothing hidden)", forced_close(), "", false };
        seq(k.g, answer);
        cases.push_back(k);
    }
    {
        kcase k{ "a closing tag after other content text (shown as before)", forced_close(), "", false };
        seq(k.g, answer);
        put(k.g, close, false);
        seq(k.g, " Done.");
        cases.push_back(k);
    }
    {
        // the model's own close, nothing forced: a second tag of its own is shown as before
        kcase k{ "the model's own close, then a second tag (nothing forced, shown as before)", {}, "", false };
        seq(k.g, think + "\n");
        put(k.g, close, false);
        seq(k.g, "\n\n");
        put(k.g, close, false);
        seq(k.g, "\n\n" + answer);
        cases.push_back(k);
    }
    const std::pair<const char *, task_response_type> routes[] = { { "chat", TASK_RESPONSE_TYPE_OAI_CHAT },
                                                                   { "anthropic", TASK_RESPONSE_TYPE_ANTHROPIC } };
    for (const kcase & k : cases) {
        bool        ok = true;
        std::string seen, bare_seen;
        try {
            for (const auto & [rname, rt] : routes) {
                for (const bool stream : { false, true }) {
                    const shown with = run(k.g, rt, stream, true);
                    const shown bare = run(k.g, rt, stream, false);
                    const bool  tagged = bare.content.rfind(tag, 0) == 0;
                    const bool  good = k.want.empty() ? with.content == bare.content && with.reasoning == bare.reasoning
                                                      : with.content == k.want && with.reasoning == bare.reasoning &&
                                                            with.content.find(tag) == std::string::npos;
                    ok = ok && good && (!k.fault || tagged) && with.reasoning.rfind(think, 0) == 0;
                    if (seen.empty() || !good) {
                        seen      = safe_json_to_str(json(with.content));
                        bare_seen = safe_json_to_str(json(bare.content));
                    }
                }
            }
        } catch (const std::exception & e) {
            ok   = false;
            seen = std::string("threw: ") + e.what();
        }
        report(ok, k.name + ": " + safe_json_to_str(json(text(k.g))) + " -> content " + seen +
                       " on the chat and Anthropic routes, streamed and not" +
                       (k.fault ? "; without the filter " + bare_seen : "; the same without the filter"));
    }
    std::printf("reasoning-check: close %s (%d failed)\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails;
}
