// fnfl-chat-check (GLFL16 thinking budget correction): the server shell's reasoning tags and chat parse for the
// model's own template, on the CPU (the vocabulary and chat template from the GGUF metadata, no weights).
//
// Tags: think_tag_texts and think_tag_ids give the opening tag, the closing tag a spent budget forces and the other
// endings Qwen3-Coder's parser accepts (C2: <tool_call>); prompt_opens_reasoning reads a rendered prompt with every
// ending. Parse (C1, R16-P2's rule): each case's request goes through oaicompat_chat_params_parse and the server's
// request schema, as the HTTP handler sends it; its text (the model's output after the generation prompt) is parsed
// once as the final reply and once streamed one UTF-8 character at a time through task_result_state, the deltas
// assembled. Both must equal the expected message.
//
// IFR1: server-reasoning-check's R1/R2 requests and its R3 UTF-8 cases, written to FIXTURE (default
// fnfl-think-utf8.txt) for `fnfl-drive think-test FIXTURE`.
//
// usage: fnfl-chat-check -m <model.gguf> [FIXTURE]. Exit status 0 when every check passes.

#include "chat.h"
#include "common.h"
#include "fn/fnfl.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"
#include "server-common.h"
#include "server-reasoning-check.h"
#include "server-schema.h"
#include "server-task.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void report(bool ok, const std::string & what, const std::string & detail = "") {
    std::printf("chat-check: %s %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(), detail.empty() ? "" : ": ",
                detail.c_str());
    g_fail += ok ? 0 : 1;
}

std::string gguf_string(const std::string & path, const char * key) {
    gguf_init_params gp = { /* no_alloc */ true, /* ctx */ nullptr };
    gguf_context *   g  = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) {
        throw std::runtime_error("cannot open GGUF " + path);
    }
    std::string     out;
    const int64_t   k = gguf_find_key(g, key);
    if (k >= 0) {
        out = gguf_get_val_str(g, k);
    }
    gguf_free(g);
    return out;
}

std::string ids_str(const std::vector<int32_t> & v) {
    std::string s;
    for (const int32_t x : v) {
        s += (s.empty() ? "" : " ") + std::to_string(x);
    }
    return s.empty() ? std::string("-") : s;
}

json message_json(const common_chat_msg & m) {
    json calls = json::array();
    for (const common_chat_tool_call & t : m.tool_calls) {
        calls.push_back({ { "name", t.name }, { "arguments", json::parse(t.arguments) } });
    }
    return { { "content", m.content }, { "reasoning_content", m.reasoning_content }, { "tool_calls", calls } };
}

struct checker {
    llama_vocab *                 vocab;
    const common_params &         params_base;
    std::vector<llama_logit_bias> logit_bias_eog;
    const server_chat_params &    opt;

    json request(bool thinking, const json & tools) const {
        json body = { { "messages", json::array({ { { "role", "user" }, { "content", "q" } } }) },
                      { "chat_template_kwargs", { { "enable_thinking", thinking } } } };
        if (!tools.empty()) {
            body["tools"] = tools;
        }
        return body;
    }

    llama_tokens prompt_of(bool thinking) const {
        json       body = request(thinking, json::array());
        const json lp   = oaicompat_chat_params_parse(body, opt);
        return common_tokenize(vocab, lp.at("prompt").get<std::string>(), false, true);
    }

    void parse(const std::string & name, bool thinking, const json & tools, const std::string & text,
               const json & want) const {
        try {
            json              body = request(thinking, tools);
            json              data = oaicompat_chat_params_parse(body, opt);
            const task_params tp   = server_schema::eval_llama_cmpl_schema(vocab, params_base, logit_bias_eog, data);
            const common_chat_parser_params & p   = tp.chat_parser_params;
            const json                        fin = message_json(common_chat_parse(text, false, p));
            task_result_state                 st(p);
            std::vector<common_chat_msg_diff> diffs;
            common_chat_msg                   streamed;
            bool                              ids  = true;
            const auto                        take = [&]() {
                for (const common_chat_msg_diff & d : diffs) {
                    streamed.content += d.content_delta;
                    streamed.reasoning_content += d.reasoning_content_delta;
                    if (d.tool_call_index != std::string::npos) {
                        if (streamed.tool_calls.size() <= d.tool_call_index) {
                            streamed.tool_calls.resize(d.tool_call_index + 1);
                        }
                        common_chat_tool_call & c = streamed.tool_calls[d.tool_call_index];
                        c.name += d.tool_call_delta.name;
                        c.arguments += d.tool_call_delta.arguments;
                        c.id = d.tool_call_delta.id.empty() ? c.id : d.tool_call_delta.id;
                    }
                }
                diffs.clear();
            };
            int32_t chunks = 0;
            for (size_t i = 0; i < text.size(); ++chunks) {
                const auto   b = static_cast<unsigned char>(text[i]);
                const size_t n = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
                st.update_chat_msg(text.substr(i, n), true, diffs);
                take();
                i += n;
            }
            st.update_chat_msg("", false, diffs);
            take();
            for (const common_chat_tool_call & t : streamed.tool_calls) {
                ids = ids && !t.id.empty();
            }
            const json str = message_json(streamed);
            const bool ok  = fin == want && str == want && ids;
            report(ok, "parse: " + name,
                   ok ? "final and " + std::to_string(chunks) + "-chunk stream as expected, " +
                            std::to_string(streamed.tool_calls.size()) + " call(s) with IDs" :
                        "final " + fin.dump() + " | streamed " + str.dump() + (ids ? "" : " (a call without an ID)") +
                            " | want " + want.dump());
        } catch (const std::exception & e) {
            report(false, "parse: " + name, std::string("threw: ") + e.what());
        }
    }
};

}  // namespace

int main(int argc, char ** argv) {
    if ((argc != 3 && argc != 4) || std::string(argv[1]) != "-m") {
        std::fprintf(stderr, "usage: fnfl-chat-check -m <model.gguf> [FIXTURE]\n");
        return 2;
    }
    const std::string fixture = argc == 4 ? argv[3] : "fnfl-think-utf8.txt";
    const std::string model_path = argv[2];
    common_log_set_verbosity_thold(-1);
    llama_vocab * vocab = llama_vocab_load_from_file(model_path.c_str());
    if (vocab == nullptr) {
        std::printf("chat-check: FAIL the vocabulary of %s\n", model_path.c_str());
        return 1;
    }
    const std::string tmpl_default  = gguf_string(model_path, "tokenizer.chat_template");
    const std::string tmpl_tool_use = gguf_string(model_path, "tokenizer.chat_template.tool_use");

    // the server's defaults for `fnfl-server --jinja` (as fnfl-chat-replay sets them: server-context.cpp load_model)
    common_params params_base;
    params_base.use_jinja = true;
    std::vector<llama_logit_bias> logit_bias_eog;
    for (llama_token t = 0; t < llama_vocab_n_tokens(vocab); t++) {
        if (llama_vocab_is_eog(vocab, t)) {
            logit_bias_eog.push_back({ t, -INFINITY });
        }
    }
    common_chat_templates_ptr tmpls = common_chat_templates_init(
        vocab, tmpl_default.empty() ? nullptr : tmpl_default.c_str(),
        tmpl_tool_use.empty() ? nullptr : tmpl_tool_use.c_str(), params_base.chat_template);
    const bool enable_thinking = params_base.use_jinja && params_base.enable_reasoning != 0 &&
                                 common_chat_templates_support_enable_thinking(tmpls.get());
    server_chat_params opt = {
        /* use_jinja             */ params_base.use_jinja,
        /* prefill_assistant     */ params_base.prefill_assistant,
        /* reasoning_format      */ params_base.reasoning_format,
        /* chat_template_kwargs  */ params_base.default_template_kwargs,
        /* tmpls                 */ std::move(tmpls),
        /* enable_thinking       */ enable_thinking,
        /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
        /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
        /* force_pure_content    */ params_base.force_pure_content_parser,
    };
    std::printf("chat-check: %s, template %zu bytes, thinking %s, reasoning format %s\n", model_path.c_str(),
                tmpl_default.size(), enable_thinking ? "on" : "off",
                common_reasoning_format_name(params_base.reasoning_format));
    const checker ck = { vocab, params_base, logit_bias_eog, opt };

    // the tags, as the server resolves them at load (server_context think_tags)
    std::string              open_tag, err;
    std::vector<std::string> end_tags;
    const bool               tags  = think_tag_texts(opt, open_tag, end_tags, err);
    const std::vector<int32_t> open  = think_tag_ids(vocab, open_tag);
    const std::vector<int32_t> close = think_tag_ids(vocab, end_tags.empty() ? std::string() : end_tags[0]);
    const std::vector<int32_t> tc    = think_tag_ids(vocab, end_tags.size() > 1 ? end_tags[1] : std::string());
    std::string                tag_list;
    for (const std::string & t : end_tags) {
        tag_list += (tag_list.empty() ? "'" : ", '") + t + "' [" + ids_str(think_tag_ids(vocab, t)) + "]";
    }
    report(tags && open == std::vector<int32_t>{ 248068 } && close == std::vector<int32_t>{ 248069 } &&
               end_tags.size() == 2 && end_tags[1] == "<tool_call>" && tc.size() == 1,
           "thinking budget: the tags, '<think>' [248068] opening, '</think>' [248069] the forced close and "
           "'<tool_call>' (one token) another ending",
           "'" + open_tag + "' [" + ids_str(open) + "]; ends " + tag_list + (tags ? "" : " " + err));
    const std::vector<std::vector<int32_t>> ends = { tc, close };
    const bool on = prompt_opens_reasoning(ck.prompt_of(true), open, ends);
    const bool off = prompt_opens_reasoning(ck.prompt_of(false), open, ends);
    const bool raw = prompt_opens_reasoning(common_tokenize(vocab, "Once upon a time", false, true), open, ends);
    report(on && !off && !raw,
           "thinking budget: the prompt opens reasoning with thinking on, not with it off or in a raw text",
           std::to_string(on) + " " + std::to_string(off) + " " + std::to_string(raw));
    // C2: a raw prompt whose reasoning a tool call ended is not open (with the close alone it would read as open)
    const llama_tokens called = common_tokenize(vocab, "<think>\nRead it.\n<tool_call>", false, true);
    const bool         c_all  = prompt_opens_reasoning(called, open, ends);
    const bool         c_one  = prompt_opens_reasoning(called, open, { close });
    report(!c_all && c_one, "thinking budget: reasoning a <tool_call> ended is closed (open with </think> alone)",
           std::to_string(c_all) + " " + std::to_string(c_one));

    // C1: the model's output in FNFL16's saved load/http/10-total.json (sha256 275e27d6...), whose content
    // came back "<think>\n" + it before the fix: the generation prompt's "<think>\n" is the parser's, not the model's
    const std::string cut = "We need answer user: \"How many prime numbers are there between 1 and 60? Answer with "
                            "the number";
    const json read_tool = json::parse(R"({"type": "function", "function": {"name": "read",
        "description": "Read a file.", "parameters": {"type": "object",
        "properties": {"path": {"type": "string"}}, "required": ["path"]}}})");
    const json tools = json::array({ read_tool });
    const std::string call = "<tool_call>\n<function=read>\n<parameter=path>\n/app/a.ts\n</parameter>\n</function>\n"
                             "</tool_call>";
    const json read_a = json::array({ { { "name", "read" }, { "arguments", { { "path", "/app/a.ts" } } } } });
    const auto msg = [](const std::string & content, const std::string & reasoning, const json & calls) {
        return json{ { "content", content }, { "reasoning_content", reasoning }, { "tool_calls", calls } };
    };
    ck.parse("C1 the saved 10-total text, cut inside its reasoning", true, json::array(), cut,
             msg("", cut, json::array()));
    ck.parse("C1 the same text with a tool offered", true, tools, cut, msg("", cut, json::array()));
    ck.parse("C1 closed reasoning, then a call", true, tools, "I should read it.\n</think>\n\n" + call,
             msg("", "I should read it.\n", read_a));
    ck.parse("C1 reasoning a call ends without </think>", true, tools, "I should read it.\n" + call,
             msg("", "I should read it.\n", read_a));
    ck.parse("C1 thinking off, an answer cut by the limit", false, json::array(),
             "There are 17 prime numbers between 1 and", msg("There are 17 prime numbers between 1 and", "",
                                                              json::array()));
    // IFR1 R1 and R2: the reasoning settings through the routes' own steps
    g_fail += reasoning_check_protocol({ vocab, params_base, logit_bias_eog, opt });
    // IFR1 R3: the UTF-8 rule with this tokenizer, the cases written for fnfl-drive think-test
    g_fail += reasoning_check_utf8({ vocab, params_base, logit_bias_eog, opt }, "qwen3.8-flash-next", open, close,
                                   { tc }, fnfl_think_utf8_entry, fixture);
    // FBH1 K2: the model's own closing tag after a forced close, through the filter and both routes' renderers
    g_fail += reasoning_check_close({ vocab, params_base, logit_bias_eog, opt }, open, close, { tc });
    llama_vocab_free(vocab);
    std::printf("chat-check: %s (%d failed)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
