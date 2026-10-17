#pragma once

#include "chat.h"
#include "chat-auto-parser.h"
#include "chat-auto-parser-helpers.h"
#include "chat-peg-parser.h"
#include "common.h"
#include "ggml.h"
#include "json-schema-to-grammar.h"
#include "json.h"

#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

using json = common_json;

// iterate over the function tools of an OpenAI-style tools array
void foreach_function(const json & tools, const std::function<void(const json &)> & fn);

// iterate over the parameters of a function tool, with the document that owns them
void foreach_parameter(const json & function, const std::function<void(const common_chat_schema_property &, const common_chat_schema_document_ptr &)> & fn);

// render a template; the override arguments let a parser feed in messages, tools or context it has rewritten
std::string common_chat_template_direct_apply_impl(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs,
    const std::optional<json> & messages_override = std::nullopt,
    const std::optional<json> & tools_override = std::nullopt,
    const std::optional<json> & additional_context = std::nullopt);

// the suffix a template appends when add_generation_prompt is set
std::string common_chat_template_generation_prompt_impl(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs,
    const std::optional<json> & messages_override = std::nullopt,
    const std::optional<json> & tools_override = std::nullopt,
    const std::optional<json> & additional_context = std::nullopt);

common_chat_params common_chat_params_init_qwen3_coder(const common_chat_template & tmpl, const autoparser::generation_params & inputs);
