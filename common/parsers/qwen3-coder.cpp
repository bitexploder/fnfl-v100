#include "parsers.h"

common_chat_params common_chat_params_init_qwen3_coder(const common_chat_template &          tmpl,
                                                              const autoparser::generation_params & inputs) {
    common_chat_params data;

    const std::string GEN_PREFIX = "<|im_start|>assistant\n";

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;

    auto supports_reasoning = tmpl.source().find("<think>") != std::string::npos;

    data.supports_thinking = supports_reasoning;
    data.preserved_tokens  = {
        "<tool_call>",
        "</tool_call>",
    };

    auto is_qwen3_coder  = !supports_reasoning;

    if (supports_reasoning) {
        data.thinking_start_tag = "<think>";
        // Support both </think> and <tool_call> as reasoning end sequences.
        // The newline variant comes first so it is included in the forced message
        // <function= is omitted, as it is a workaround for Qwen3-Coder which is not a thinking model
        data.thinking_end_tags = { "\n</think>", "</think>", "<tool_call>" };
        data.preserved_tokens.insert(data.preserved_tokens.end(), { "<think>", "</think>" });
    }

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "<|im_start|>assistant"             },
        { COMMON_CHAT_ROLE_TOOL,      "<|im_start|>user\n<tool_response>" }, // Qwen3-Coder, Qwen3.5, Nemotron Nano 3
        { COMMON_CHAT_ROLE_TOOL,      "<|im_start|>tool_response"         }, // StepFun-3.5-Flash
        { COMMON_CHAT_ROLE_USER,      "<|im_start|>user"                  },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|im_start|>system"                },
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = GEN_PREFIX;
        if (supports_reasoning) {
            data.generation_prompt += "<think>\n" + msg.reasoning_content;
            if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
                data.generation_prompt += "\n</think>\n\n";
            }
        }
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    std::vector<std::string> tool_call_starts = { "<tool_call>" };

    if (is_qwen3_coder) {
        // Match complete <function=name> opener for Qwen3-Coder models that occasionally omit the
        // starting <tool_call>. The model may hallucinate a tool name, but it is preferable over
        // constraining on <function which may occur in valid content generation, e.g. #include <functional>
        foreach_function(inputs.tools, [&](const json & tool) {
            const std::string name = tool.at("function").at("name");
            tool_call_starts.push_back("<function=" + name + ">");
        });
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(GEN_PREFIX);

        auto reasoning = p.eps();
        if (supports_reasoning && extract_reasoning) {
            reasoning = p.optional("<think>" + p.space() +
                                   p.reasoning(p.until_one_of({ "</think>", "<tool_call>" })) +
                                   (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));
        }

        // Response format parser
        if (has_response_format) {
            return generation_prompt + (reasoning << p.content(p.schema(p.json(), "response-format", inputs.json_schema)));
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto arg_close  = p.tool_arg_close(p.literal("\n</parameter>\n"));
            auto arg_string = p.rule("xml-arg-string",
                p.ac(p.tool_arg_string_value(p.until("\n</parameter>\n")) + arg_close, "\n</parameter>\n"));

            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");

                std::vector<common_peg_parser> required_args;
                std::vector<common_peg_parser> optional_args;

                foreach_parameter(function, [&](const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
                    auto rule_name = "tool-" + name + "-arg-" + param.name;

                    auto arg_open = p.tool_arg_open("<parameter=" + p.tool_arg_name(p.literal(param.name)) + ">\n");

                    auto types = param.schema->value_types();

                    auto arg_value = p.eps();
                    if (!types.has(common_chat_schema::TYPE_STRING)) {
                        arg_value = p.tool_arg_json_value(p.schema(p.json(), rule_name + "-schema", doc, *param.schema)) + arg_close;
                    } else if (types.is_only(common_chat_schema::TYPE_STRING)) {
                        arg_value = arg_string;
                    } else {
                        // The string alternative accepts any text, so the grammar only keeps the raw string
                        // rule. The parser still tries the JSON alternatives first to type the value.
                        auto json_value = p.choice();
                        if (types.has(common_chat_schema::TYPE_OBJECT)) {
                            json_value |= p.json_object();
                        }
                        if (types.has(common_chat_schema::TYPE_ARRAY)) {
                            json_value |= p.json_array();
                        }
                        if (types.has(common_chat_schema::TYPE_NUMBER) || types.has(common_chat_schema::TYPE_INTEGER)) {
                            json_value |= p.json_number();
                        }
                        if (types.has(common_chat_schema::TYPE_BOOLEAN)) {
                            json_value |= p.json_bool();
                        }
                        if (types.has(common_chat_schema::TYPE_NULL)) {
                            json_value |= p.json_null();
                        }
                        arg_value = p.gbnf(p.atomic(p.tool_arg_json_value(json_value) + arg_close) | arg_string, "xml-arg-string");
                    }

                    auto arg_rule = p.rule(rule_name, p.tool_arg(arg_open + arg_value));

                    (param.required ? required_args : optional_args).push_back(arg_rule);
                });

                // Accept required arguments in any order, as Qwen does not always adhere to the
                // order provided.
                auto args = p.permute("tool-" + name + "-args", required_args);
                if (!optional_args.empty()) {
                    args = args + p.zero_or_more(p.choice(optional_args));
                }

                auto func = p.tool(p.tool_open("<function=" + p.tool_name(p.literal(name)) + ">\n") +
                                   p.tool_args(args) +
                                   p.tool_close(p.literal("</function>\n")));

                tool_choice |= p.rule("tool-" + name, func);
            });

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;

            auto tool_call_body = tool_choice + "</tool_call>" + p.space();
            auto tool_call      = p.rule("tool-call", "<tool_call>\n" + tool_call_body);

            // Qwen3-Coder models may occasionally omit the <tool_call> token.
            auto tool_call_first = is_qwen3_coder ?
                p.rule("tool-call-first", p.optional(p.literal("<tool_call>\n")) + tool_call_body) :
                tool_call;

            auto calls      = inputs.parallel_tool_calls ? tool_call_first + p.zero_or_more(tool_call) : tool_call_first;
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(calls, min_calls, 1));

            return generation_prompt +
                   (reasoning << p.content(p.until_one_of(tool_call_starts)) << tool_calls);
        }

        // Content only parser
        return generation_prompt + (reasoning << p.content(p.rest()));
    });

    // FNFL (TC1): the server parses with a lenient copy of the parser above; the strict one still builds the grammar.
    // FNFL's sampler applies no grammar, so the model's tool-call text reaches the parser unconstrained. A call the
    // strict rule rejects (arguments in any order, whitespace and closing-tag variants, an unknown function or
    // parameter, a missing argument, a value of the wrong type, a call cut off by the end of generation) is still
    // delivered as a tool call with its arguments as written, and text outside the calls stays reasoning or content.
    // Every node the lenient rules match carries the LENIENT tag, which common_chat_peg_parse reports.
    const bool lenient_tools = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && !has_response_format;
    auto lenient = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        const auto LENIENT = common_chat_peg_builder::LENIENT;
        const bool think   = supports_reasoning && extract_reasoning;

        auto reasoning = p.eps();
        if (think) {
            reasoning = p.optional("<think>" + p.space() +
                                   p.reasoning(p.until_one_of({ "</think>", "<tool_call>" })) +
                                   (p.literal("</think>") | p.peek(p.literal("<tool_call>")) | p.end()));
        }

        // the strict call rule, unchanged, so a call it accepts parses exactly as before
        auto arg_close_strict = p.tool_arg_close(p.literal("\n</parameter>\n"));
        auto arg_string_strict = p.rule("xml-arg-string",
            p.ac(p.tool_arg_string_value(p.until("\n</parameter>\n")) + arg_close_strict, "\n</parameter>\n"));

        // a value ends at the </parameter> that is followed by the next parameter, the call's end or the text's end,
        // so a value may itself contain </parameter> text
        const std::vector<std::string> value_ends = { "\n</parameter>", "</parameter>" };
        auto value_close = p.rule("tc1-value-close", p.optional(p.literal("\n")) + "</parameter>" + p.space() +
            (p.literal("<parameter=") | p.literal("</function>") | p.literal("</tool_call>") | p.end()));
        auto string_value = p.rule("tc1-string-value", p.until_one_of(value_ends) +
            p.zero_or_more(p.negate(value_close) + (p.literal("\n</parameter>") | p.literal("</parameter>")) +
                           p.until_one_of(value_ends)));
        auto string_close = p.tool_arg_close((p.optional(p.literal("\n")) + p.literal("</parameter>")) | p.end());
        // any parameter name, its raw text as a string
        auto any_arg = p.rule("tc1-arg", p.tool_arg(
            p.tool_arg_open("<parameter=" + p.tool_arg_name(p.chars("[^<>\\n]")) + ">") + p.optional(p.literal("\n")) +
            p.tool_arg_string_value(string_value) + string_close));
        // text inside a call that is not an argument is dropped
        auto junk = p.space() + p.tag(LENIENT, p.negate(p.literal("</function>") | p.literal("</tool_call>")) + p.any() +
                                                   p.until_one_of({ "<parameter=", "</function>", "</tool_call>" }));
        auto call_close = p.space() +
            p.tool_close(p.literal("</function>") | p.peek(p.literal("</tool_call>")) | p.end());

        auto strict_choice  = p.choice();
        auto lenient_choice = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");

            std::vector<common_peg_parser> required_args;
            std::vector<common_peg_parser> optional_args;
            auto typed_args = p.choice();
            foreach_parameter(function, [&](const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
                auto rule_name = "tool-" + name + "-arg-" + param.name;
                auto arg_open  = p.tool_arg_open("<parameter=" + p.tool_arg_name(p.literal(param.name)) + ">\n");
                auto types     = param.schema->value_types();

                auto arg_value  = p.eps();
                auto json_value = p.choice();
                if (!types.has(common_chat_schema::TYPE_STRING)) {
                    arg_value  = p.tool_arg_json_value(p.schema(p.json(), rule_name + "-schema", doc, *param.schema)) + arg_close_strict;
                    json_value = p.json();
                } else if (types.is_only(common_chat_schema::TYPE_STRING)) {
                    arg_value = arg_string_strict;
                } else {
                    if (types.has(common_chat_schema::TYPE_OBJECT)) {
                        json_value |= p.json_object();
                    }
                    if (types.has(common_chat_schema::TYPE_ARRAY)) {
                        json_value |= p.json_array();
                    }
                    if (types.has(common_chat_schema::TYPE_NUMBER) || types.has(common_chat_schema::TYPE_INTEGER)) {
                        json_value |= p.json_number();
                    }
                    if (types.has(common_chat_schema::TYPE_BOOLEAN)) {
                        json_value |= p.json_bool();
                    }
                    if (types.has(common_chat_schema::TYPE_NULL)) {
                        json_value |= p.json_null();
                    }
                    arg_value = p.gbnf(p.atomic(p.tool_arg_json_value(json_value) + arg_close_strict) | arg_string_strict, "xml-arg-string");
                }
                auto arg_rule = p.rule(rule_name, p.tool_arg(arg_open + arg_value));
                (param.required ? required_args : optional_args).push_back(arg_rule);

                // lenient: a typed value where the text parses as that JSON, else the string rule (any_arg)
                if (!types.is_only(common_chat_schema::TYPE_STRING)) {
                    typed_args |= p.rule("tc1-" + rule_name, p.tool_arg(
                        p.tool_arg_open("<parameter=" + p.tool_arg_name(p.literal(param.name)) + ">") + p.space() +
                        p.tool_arg_json_value(json_value) + p.space() +
                        ((p.peek(value_close) + p.tool_arg_close(p.literal("</parameter>"))) | p.tool_arg_close(p.end()))));
                }
            });

            auto args = p.permute("tool-" + name + "-args", required_args);
            if (!optional_args.empty()) {
                args = args + p.zero_or_more(p.choice(optional_args));
            }
            strict_choice |= p.rule("tool-" + name, p.tool(p.tool_open("<function=" + p.tool_name(p.literal(name)) + ">\n") +
                                                           p.tool_args(args) +
                                                           p.tool_close(p.literal("</function>\n"))));

            auto lenient_args = p.zero_or_more((p.space() + (typed_args | any_arg)) | junk);
            lenient_choice |= p.rule("tc1-tool-" + name, p.tool(p.tool_open("<function=" + p.tool_name(p.literal(name)) + ">") +
                                                                 p.tool_args(lenient_args) + call_close));
        });
        // an unknown function name is still a call
        lenient_choice |= p.rule("tc1-tool-any", p.tool(p.tool_open("<function=" + p.tool_name(p.chars("[^<>\\n]")) + ">") +
                                                        p.tool_args(p.zero_or_more((p.space() + any_arg) | junk)) + call_close));

        auto strict_call  = p.rule("tool-call", "<tool_call>\n" + strict_choice + "</tool_call>" + p.space());
        auto lenient_call = p.rule("tc1-tool-call", p.tag(LENIENT, "<tool_call>" + p.space() + lenient_choice + p.space() +
                                                                   p.optional(p.literal("</tool_call>")) + p.space()));
        // outside a call: text after a call stays content, a stray </think> is dropped, a <tool_call> that opens no
        // function is text
        std::vector<std::string> text_ends = { "<tool_call>" };
        auto item = strict_call | lenient_call;
        if (think) {
            text_ends.push_back("</think>");
            item |= p.tag(LENIENT, p.literal("</think>") + p.space());
        }
        item |= p.tag(LENIENT, p.content(p.literal("<tool_call>")));
        item |= p.tag(LENIENT, p.content(p.until_one_of(text_ends)));

        return p.literal(GEN_PREFIX) + reasoning + p.space() + p.content(p.until_one_of(text_ends)) +
               p.zero_or_more(item) + p.end();
    });

    data.parser = lenient_tools ? lenient.save() : parser.save();

    if (include_grammar) {
        data.grammar_lazy = has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });

        if (data.grammar_lazy) {
            for (const auto & start : tool_call_starts) {
                data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, start });
            }
        }
    }

    return data;
}
