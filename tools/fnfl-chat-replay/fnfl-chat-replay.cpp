// fnfl-chat-replay: replay raw generated text through the server's chat parsing and streaming code (TC1).
//
// For each case (a chat request plus the raw text the model generated), it builds the task the server would build
// (oaicompat_chat_params_parse, then the server's request schema), feeds the text one token at a time the way the
// slot's process_token does (UTF-8 hold-back, stop strings, special tokens rendered only when preserved), and passes
// each piece through task_result_state::update_chat_msg and the OpenAI chat stream JSON, as the HTTP handler does.
// It then runs the final parse, and the same text as one non-streamed reply.
//
// Reported per case: any throw (a stream error to the client), any retraction (the server's message no longer
// extends what the client was sent), whether the concatenated deltas equal the final message, whether the streamed
// and non-streamed replies agree, the case's expectations, and the server's TC1 warning lines.
// CPU only: it loads the model's vocabulary and chat template from the GGUF metadata, no weights.
//
// usage: fnfl-chat-replay -m <model.gguf> [--json <out.jsonl>] [--quiet] <cases.json|cases.jsonl>...
// A case: {"name", "request" | "request_file", "text" | "tokens", "stop": "eos"|"limit", "expect": {...}}
//   expect: {"content", "reasoning_content", "tool_calls": [{"name", "arguments": {...}}], "lenient": bool}
// Exit status: 0 when every case passes.

#include "chat.h"
#include "common.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"
#include "server-common.h"
#include "server-schema.h"
#include "server-task.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct client_view {
    std::string reasoning;
    std::string content;
    std::vector<common_chat_tool_call> calls;
    std::string finish_reason;
};

std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot read " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string dir_of(const std::string & path) {
    const size_t p = path.find_last_of('/');
    return p == std::string::npos ? std::string(".") : path.substr(0, p);
}

std::string gguf_string(const std::string & path, const char * key) {
    gguf_init_params gp = { /* no_alloc */ true, /* ctx */ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) {
        throw std::runtime_error("cannot open GGUF " + path);
    }
    std::string out;
    const int64_t k = gguf_find_key(g, key);
    if (k >= 0) {
        out = gguf_get_val_str(g, k);
    }
    gguf_free(g);
    return out;
}

// apply one stream chunk's delta the way an OpenAI client accumulates it
void apply_chunk(client_view & cv, const json & chunk, std::vector<std::string> & notes) {
    if (!chunk.contains("choices") || chunk.at("choices").empty()) {
        return;
    }
    const json & choice = chunk.at("choices").at(0);
    if (choice.contains("finish_reason") && !choice.at("finish_reason").is_null()) {
        cv.finish_reason = choice.at("finish_reason").get<std::string>();
    }
    if (!choice.contains("delta")) {
        return;
    }
    const json & d = choice.at("delta");
    if (d.contains("reasoning_content") && d.at("reasoning_content").is_string()) {
        cv.reasoning += d.at("reasoning_content").get<std::string>();
    }
    if (d.contains("content") && d.at("content").is_string()) {
        cv.content += d.at("content").get<std::string>();
    }
    if (d.contains("tool_calls")) {
        for (const auto & tc : d.at("tool_calls")) {
            const size_t idx = tc.at("index").get<size_t>();
            if (idx > cv.calls.size()) {
                notes.push_back("tool call index " + std::to_string(idx) + " skips ahead");
            }
            while (cv.calls.size() <= idx) {
                cv.calls.emplace_back();
            }
            auto & c = cv.calls[idx];
            if (tc.contains("id")) {
                const auto id = tc.at("id").get<std::string>();
                if (!c.id.empty() && c.id != id) {
                    notes.push_back("tool call " + std::to_string(idx) + " id changed");
                }
                c.id = id;
            }
            if (tc.contains("function")) {
                const json & fn = tc.at("function");
                if (fn.contains("name")) {
                    const auto name = fn.at("name").get<std::string>();
                    if (!c.name.empty() && c.name != name) {
                        notes.push_back("tool call " + std::to_string(idx) + " renamed '" + c.name + "' -> '" + name + "'");
                    }
                    c.name = name;
                }
                if (fn.contains("arguments")) {
                    c.arguments += fn.at("arguments").get<std::string>();
                }
            }
        }
    }
}

void apply_json(client_view & cv, const json & j, std::vector<std::string> & notes) {
    if (j.is_array()) {
        for (const auto & c : j) {
            apply_chunk(cv, c, notes);
        }
    } else if (j.is_object()) {
        apply_chunk(cv, j, notes);
    }
}

bool starts_with(const std::string & s, const std::string & prefix) {
    return s.compare(0, prefix.size(), prefix) == 0;
}

// the client must only ever hold a prefix of what the server now believes
std::string retraction(const client_view & cv, const common_chat_msg & m) {
    if (!starts_with(m.reasoning_content, cv.reasoning)) {
        return "reasoning retracted";
    }
    if (!starts_with(m.content, cv.content)) {
        return "content retracted";
    }
    if (m.tool_calls.size() < cv.calls.size()) {
        return "tool call withdrawn (" + std::to_string(cv.calls.size()) + " sent, " +
               std::to_string(m.tool_calls.size()) + " now)";
    }
    for (size_t i = 0; i < cv.calls.size(); i++) {
        const auto & s = cv.calls[i];
        const auto & n = m.tool_calls[i];
        if ((!s.name.empty() && s.name != n.name) || (!s.id.empty() && s.id != n.id) ||
            !starts_with(n.arguments, s.arguments)) {
            return "tool call " + std::to_string(i) + " changed after it was sent";
        }
    }
    return "";
}

json msg_json(const common_chat_msg & m) {
    json calls = json::array();
    for (const auto & c : m.tool_calls) {
        calls.push_back({ { "name", c.name }, { "id", c.id }, { "arguments", c.arguments } });
    }
    return { { "reasoning_content", m.reasoning_content }, { "content", m.content }, { "tool_calls", calls } };
}

json view_json(const client_view & cv) {
    json calls = json::array();
    for (const auto & c : cv.calls) {
        calls.push_back({ { "name", c.name }, { "id", c.id }, { "arguments", c.arguments } });
    }
    return { { "reasoning_content", cv.reasoning }, { "content", cv.content }, { "tool_calls", calls },
             { "finish_reason", cv.finish_reason } };
}

// partial stop-string search, as the slot does (server-context.cpp find_stopping_strings)
size_t find_stop(const std::vector<std::string> & words, const std::string & text, size_t last_token_size,
                 bool is_full_stop) {
    size_t stop_pos = std::string::npos;
    for (const auto & word : words) {
        size_t pos;
        if (is_full_stop) {
            const size_t tmp      = word.size() + last_token_size;
            const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
            pos                   = text.find(word, from_pos);
        } else {
            pos = string_find_partial_stop(text, word);
        }
        if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
            stop_pos = pos;
        }
    }
    return stop_pos;
}

struct log_tail {
    std::string path;
    size_t      offset = 0;

    std::vector<std::string> take() {
        common_log_flush(common_log_main());
        std::vector<std::string> out;
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return out;
        }
        f.seekg((std::streamoff) offset);
        std::string line;
        while (std::getline(f, line)) {
            offset += line.size() + 1;
            if (line.find("TC1") != std::string::npos || line.find("unparsed") != std::string::npos) {
                out.push_back(line);
            }
        }
        return out;
    }
};

}  // namespace

int main(int argc, char ** argv) {
    std::string              model_path;
    std::string              json_out;
    bool                     quiet = false;
    std::vector<std::string> case_files;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if ((a == "-m" || a == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "--json" && i + 1 < argc) {
            json_out = argv[++i];
        } else if (a == "--quiet") {
            quiet = true;
        } else {
            case_files.push_back(a);
        }
    }
    if (model_path.empty() || case_files.empty()) {
        fprintf(stderr, "usage: %s -m <model.gguf> [--json <out.jsonl>] [--quiet] <cases.json|.jsonl>...\n", argv[0]);
        return 2;
    }

    log_tail logs;
    logs.path = json_out.empty() ? std::string("fnfl-chat-replay.log") : json_out + ".log";
    common_log_set_file(common_log_main(), logs.path.c_str());
    common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);

    llama_vocab * vocab = llama_vocab_load_from_file(model_path.c_str());
    if (vocab == nullptr) {
        fprintf(stderr, "cannot load the vocabulary of %s\n", model_path.c_str());
        return 2;
    }
    const std::string tmpl_default  = gguf_string(model_path, "tokenizer.chat_template");
    const std::string tmpl_tool_use = gguf_string(model_path, "tokenizer.chat_template.tool_use");

    // the server's defaults for `llama-server --jinja` (server-context.cpp load_model and init)
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
    server_chat_params chat_params = {
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

    FILE * jout = json_out.empty() ? nullptr : fopen(json_out.c_str(), "w");
    int    n_pass = 0, n_fail = 0;

    for (const auto & cf : case_files) {
        std::vector<json> cases;
        {
            const std::string raw = read_file(cf);
            try {
                json j = json::parse(raw);
                if (j.is_array()) {
                    for (auto & c : j) {
                        cases.push_back(c);
                    }
                } else {
                    cases.push_back(j);
                }
            } catch (const std::exception &) {
                std::istringstream ls(raw);
                std::string        line;
                while (std::getline(ls, line)) {
                    if (!line.empty()) {
                        cases.push_back(json::parse(line));
                    }
                }
            }
        }

        for (auto & c : cases) {
            const std::string name = c.value("name", cf);
            std::vector<std::string> fails;
            std::vector<std::string> notes;
            json                     rec = { { "name", name }, { "file", cf } };
            logs.take();

            json body = c.contains("request") ? c.at("request")
                                              : json::parse(read_file(dir_of(cf) + "/" + c.at("request_file").get<std::string>()));
            body["stream"] = true;

            // a recorded turn: the request's messages end with the assistant message the model wrote; the text is
            // that message re-rendered by the chat template, and the parse must give the message back
            const bool turn = c.value("turn", false);
            if (turn) {
                json         msgs = json::array();
                const json & all  = body.at("messages");
                const size_t n    = c.contains("turn_index") ? c.at("turn_index").get<size_t>() + 1 : all.size();
                for (size_t i = 0; i < n; i++) {
                    msgs.push_back(all.at(i));
                }
                const json last = msgs.back();
                json       e    = { { "content", last.value("content", json()).is_string() ? last.at("content") : json("") },
                                    { "reasoning_content", last.value("reasoning_content", std::string()) },
                                    { "tool_calls", json::array() } };
                for (const auto & tc : last.value("tool_calls", json::array())) {
                    e["tool_calls"].push_back({ { "name", tc.at("function").at("name") },
                                                { "arguments", json::parse(tc.at("function").at("arguments").get<std::string>()) } });
                }
                c["expect"] = e;
                auto parsed = common_chat_msgs_parse_oaicompat(msgs);
                const common_chat_msg new_msg = parsed.back();
                parsed.pop_back();
                c["text"] = common_chat_format_single(chat_params.tmpls.get(), parsed, new_msg, false, true);
                json past = json::array();
                for (size_t i = 0; i + 1 < n; i++) {
                    past.push_back(msgs.at(i));
                }
                body["messages"] = past;
            }

            task_params tp;
            try {
                json data = oaicompat_chat_params_parse(body, chat_params);
                tp        = server_schema::eval_llama_cmpl_schema(vocab, params_base, logit_bias_eog, data);
            } catch (const std::exception & e) {
                fails.push_back(std::string("request rejected: ") + e.what());
            }

            if (turn && fails.empty()) {
                // the rendered message is the generation prompt, the generated text, then the end-of-turn token
                std::string       text = c.at("text").get<std::string>();
                const std::string gp   = tp.chat_parser_params.generation_prompt;
                const size_t      eot  = text.rfind("<|im_end|>");
                if (text.compare(0, gp.size(), gp) != 0 || eot == std::string::npos) {
                    fails.push_back("the rendered turn does not start with the generation prompt or lacks <|im_end|>");
                } else {
                    c["text"] = text.substr(gp.size(), eot - gp.size());
                }
            }

            std::vector<llama_token> toks;
            if (fails.empty()) {
                if (c.contains("tokens")) {
                    toks = c.at("tokens").get<std::vector<llama_token>>();
                } else {
                    toks = common_tokenize(vocab, c.at("text").get<std::string>(), false, true);
                }
            }
            const bool cut_off = c.value("stop", std::string("eos")) == "limit";

            // the streamed reply
            client_view       cv;
            std::string       generated;
            std::string       retract;
            task_result_state st(tp.chat_parser_params);
            bool              threw = false;
            int               n_partials = 0;
            if (fails.empty()) {
                try {
                    {
                        server_task_result_cmpl_partial begin;
                        begin.is_begin = true;
                        begin.update(st);
                    }
                    size_t n_sent = 0;
                    int    n_gen  = 0;
                    bool   stopped_word = false;
                    for (size_t i = 0; i < toks.size() && !stopped_word; i++) {
                        const llama_token tok = toks[i];
                        n_gen++;
                        const std::string piece =
                            common_token_to_piece(vocab, tok, tp.sampling.preserved_tokens.count(tok) > 0);
                        generated += piece;
                        if (validate_utf8(generated) < generated.size()) {
                            continue;  // incomplete UTF-8: held back
                        }
                        size_t            pos      = std::min(n_sent, generated.size());
                        const std::string str_test = generated.substr(pos);
                        bool              send     = true;
                        size_t stop_pos = find_stop(tp.antiprompt, str_test, piece.size(), true);
                        if (stop_pos != std::string::npos) {
                            generated.erase(generated.begin() + pos + stop_pos, generated.end());
                            pos          = std::min(n_sent, generated.size());
                            stopped_word = true;
                        } else if (!llama_vocab_is_eog(vocab, tok)) {
                            stop_pos = find_stop(tp.antiprompt, str_test, piece.size(), false);
                            send     = stop_pos == std::string::npos;
                        }
                        std::string to_send;
                        if (send) {
                            to_send = generated.substr(pos);
                            n_sent += to_send.size();
                        }
                        server_task_result_cmpl_partial r;
                        r.content           = to_send;
                        r.tokens            = { tok };
                        r.n_decoded         = n_gen;
                        r.n_prompt_tokens   = 0;
                        r.n_prompt_tokens_cache = 0;
                        r.post_sampling_probs   = false;
                        r.res_type          = TASK_RESPONSE_TYPE_OAI_CHAT;
                        r.oaicompat_model   = "replay";
                        r.oaicompat_cmpl_id = "chatcmpl-replay";
                        r.update(st);
                        apply_json(cv, r.to_json(), notes);
                        n_partials++;
                        if (retract.empty()) {
                            retract = retraction(cv, st.chat_msg);
                        }
                    }
                    server_task_result_cmpl_final f;
                    f.content           = "";
                    f.stream            = true;
                    f.include_usage     = false;
                    f.truncated         = false;
                    f.n_decoded         = n_gen;
                    f.n_prompt_tokens   = 0;
                    f.n_prompt_tokens_cache = 0;
                    f.n_tokens_cached   = 0;
                    f.has_new_line      = false;
                    f.post_sampling_probs = false;
                    f.stop              = cut_off ? STOP_TYPE_LIMIT : (stopped_word ? STOP_TYPE_WORD : STOP_TYPE_EOS);
                    f.res_type          = TASK_RESPONSE_TYPE_OAI_CHAT;
                    f.oaicompat_model   = "replay";
                    f.oaicompat_cmpl_id = "chatcmpl-replay";
                    f.update(st);
                    apply_json(cv, f.to_json(), notes);
                    if (retract.empty()) {
                        retract = retraction(cv, st.chat_msg);
                    }
                } catch (const std::exception & e) {
                    threw = true;
                    fails.push_back(std::string("stream THROW after ") + std::to_string(n_partials) +
                                    " partials: " + e.what());
                }
            }
            if (!retract.empty()) {
                fails.push_back("retraction: " + retract);
            }

            // the same text as one non-streamed reply
            common_chat_msg nonstream;
            bool            ns_threw = false;
            if (fails.empty() || threw) {
                try {
                    task_result_state             st2(tp.chat_parser_params);
                    server_task_result_cmpl_final f;
                    f.content           = generated;
                    f.stream            = false;
                    f.include_usage     = false;
                    f.truncated         = false;
                    f.n_decoded         = (int32_t) toks.size();
                    f.n_prompt_tokens   = 0;
                    f.n_prompt_tokens_cache = 0;
                    f.n_tokens_cached   = 0;
                    f.has_new_line      = false;
                    f.post_sampling_probs = false;
                    f.stop              = cut_off ? STOP_TYPE_LIMIT : STOP_TYPE_EOS;
                    f.res_type          = TASK_RESPONSE_TYPE_OAI_CHAT;
                    f.oaicompat_model   = "replay";
                    f.oaicompat_cmpl_id = "chatcmpl-replay";
                    f.update(st2);
                    (void) f.to_json();
                    nonstream = st2.chat_msg;
                } catch (const std::exception & e) {
                    ns_threw = true;
                    fails.push_back(std::string("non-stream THROW: ") + e.what());
                }
            }

            const common_chat_msg & fin = st.chat_msg;
            if (!threw) {
                // the concatenated deltas equal the final message
                if (cv.reasoning != fin.reasoning_content) {
                    fails.push_back("deltas != final: reasoning");
                }
                if (cv.content != fin.content) {
                    fails.push_back("deltas != final: content");
                }
                if (cv.calls.size() != fin.tool_calls.size()) {
                    fails.push_back("deltas != final: " + std::to_string(cv.calls.size()) + " calls streamed, " +
                                    std::to_string(fin.tool_calls.size()) + " final");
                } else {
                    for (size_t i = 0; i < cv.calls.size(); i++) {
                        if (cv.calls[i].name != fin.tool_calls[i].name || cv.calls[i].id != fin.tool_calls[i].id ||
                            cv.calls[i].arguments != fin.tool_calls[i].arguments) {
                            fails.push_back("deltas != final: tool call " + std::to_string(i));
                        }
                    }
                }
                for (size_t i = 0; i < cv.calls.size(); i++) {
                    json a = json::parse_no_throw(cv.calls[i].arguments);
                    if (a.is_discarded() || !a.is_object()) {
                        fails.push_back("tool call " + std::to_string(i) + " arguments are not a JSON object: " +
                                        cv.calls[i].arguments);
                    }
                }
                if (!ns_threw && (nonstream.reasoning_content != fin.reasoning_content || nonstream.content != fin.content ||
                                  nonstream.tool_calls.size() != fin.tool_calls.size())) {
                    fails.push_back("streamed and non-streamed replies differ");
                } else if (!ns_threw) {
                    for (size_t i = 0; i < fin.tool_calls.size(); i++) {
                        if (nonstream.tool_calls[i].name != fin.tool_calls[i].name ||
                            nonstream.tool_calls[i].arguments != fin.tool_calls[i].arguments) {
                            fails.push_back("streamed and non-streamed tool call " + std::to_string(i) + " differ");
                        }
                    }
                }
                const std::string want_finish = fin.tool_calls.empty() ? (cut_off ? "length" : "stop")
                                                                       : (cut_off ? "length" : "tool_calls");
                if (cv.finish_reason != want_finish) {
                    fails.push_back("finish_reason " + cv.finish_reason + ", expected " + want_finish);
                }
                // an opened call never reaches the client as text (unless the case expects that text)
                const bool want_text = c.contains("expect") && c.at("expect").contains("content");
                if (!want_text && (fin.content.find("<function=") != std::string::npos ||
                                   fin.reasoning_content.find("<function=") != std::string::npos)) {
                    fails.push_back("call markup left in reasoning or content");
                }
            }
            for (const auto & n : notes) {
                fails.push_back("stream: " + n);
            }

            // expectations
            if (c.contains("expect") && !threw) {
                const json & e = c.at("expect");
                if (e.contains("content") && e.at("content").get<std::string>() != fin.content) {
                    fails.push_back("expected content " + e.at("content").dump() + ", got " + json(fin.content).dump());
                }
                if (e.contains("reasoning_content") &&
                    e.at("reasoning_content").get<std::string>() != fin.reasoning_content) {
                    fails.push_back("expected reasoning " + e.at("reasoning_content").dump() + ", got " +
                                    json(fin.reasoning_content).dump());
                }
                if (e.contains("tool_calls")) {
                    const json & ec = e.at("tool_calls");
                    if (ec.size() != fin.tool_calls.size()) {
                        fails.push_back("expected " + std::to_string(ec.size()) + " tool calls, got " +
                                        std::to_string(fin.tool_calls.size()));
                    } else {
                        for (size_t i = 0; i < ec.size(); i++) {
                            const auto & tc = fin.tool_calls[i];
                            json         a  = json::parse_no_throw(tc.arguments);
                            if (ec[i].at("name").get<std::string>() != tc.name || a.is_discarded() ||
                                a != ec[i].at("arguments")) {
                                fails.push_back("tool call " + std::to_string(i) + ": expected " +
                                                ec[i].at("name").get<std::string>() + " " + ec[i].at("arguments").dump() +
                                                ", got " + tc.name + " " + tc.arguments);
                            }
                        }
                    }
                }
            }

            const auto warnings = logs.take();
            int        n_guard  = 0;
            int        n_lenient = 0;
            for (const auto & w : warnings) {
                n_guard += w.find("TC1 stream guard") != std::string::npos;
                n_lenient += w.find("TC1 lenient") != std::string::npos;
            }
            if (n_guard > 0) {
                fails.push_back("stream guard used " + std::to_string(n_guard) + " time(s)");
            }
            if (c.contains("expect") && c.at("expect").contains("lenient")) {
                const bool want = c.at("expect").at("lenient").get<bool>();
                if (want != (n_lenient > 0)) {
                    fails.push_back(std::string("expected lenient warning ") + (want ? "yes" : "no") + ", got " +
                                    std::to_string(n_lenient));
                }
            }

            const bool pass = fails.empty();
            (pass ? n_pass : n_fail)++;
            if (!quiet || !pass) {
                printf("%s %s (%zu tokens, %d partials, %zu calls)\n", pass ? "PASS" : "FAIL", name.c_str(),
                       toks.size(), n_partials, fin.tool_calls.size());
                for (const auto & f : fails) {
                    printf("    %s\n", f.c_str());
                }
            }
            if (jout) {
                rec["pass"]      = pass;
                rec["fails"]     = fails;
                rec["tokens"]    = toks.size();
                rec["final"]     = msg_json(fin);
                rec["client"]    = view_json(cv);
                rec["nonstream"] = msg_json(nonstream);
                rec["warnings"]  = warnings;
                rec["chat_format"] = common_chat_format_name(tp.chat_parser_params.format);
                fprintf(jout, "%s\n", rec.dump().c_str());
            }
        }
    }
    if (jout) {
        fclose(jout);
    }
    printf("fnfl-chat-replay: %d PASS, %d FAIL\n", n_pass, n_fail);
    llama_vocab_free(vocab);
    return n_fail == 0 ? 0 : 1;
}
