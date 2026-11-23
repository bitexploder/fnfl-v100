#pragma once

// IFR1: CPU checks of the server shell's reasoning settings, linked by each engine's chat check (fnfl-chat-check,
// dsfl-chat-check, glfl-chat-check), never by the server. Each prints "reasoning-check: PASS|FAIL ..." lines and
// returns its failures.

#include "common.h"
#include "llama.h"
#include "server-common.h"

#include <string>
#include <vector>

struct reasoning_check_ctx {
    const llama_vocab *                   vocab;
    const common_params &                 params_base;
    const std::vector<llama_logit_bias> & logit_bias_eog;
    server_chat_params &                  opt; // its reasoning_budget: set to the cases' server default, then restored
};

// R1, R2: requests through the routes' own steps (the Anthropic translation, the chat parse and the completion
// schema), each exception mapped as its route maps it: status, effective budget and error against each case
int reasoning_check_protocol(const reasoning_check_ctx & c);

// FBH1 K2: replies built from the model's real tokens where the model writes its own closing tag after a forced close
// (the newline, the closing sequence and the blank line forced, as K1 feeds them), each through the shell's filter
// (think_close_hide, as apply_step uses it) and then the result classes of the chat and Anthropic routes, streamed and
// not: the content is the answer alone. close: the shell's closing sequence
int reasoning_check_close(const reasoning_check_ctx & c, const std::vector<int32_t> & open,
                          const std::vector<int32_t> & close, const std::vector<std::vector<int32_t>> & ends);

// R3: the server's UTF-8 table (think_utf8_table with entry, the engine's fnfl_think_utf8_entry) against the check's
// own decoder over the whole vocabulary; then cases built from the model's real token pieces where the budget lands
// inside a 2, 3 or 4-byte character, their tokens worked out from the bytes (the rule) and as the old tracker gives
// them, each text through the request's parser final and streamed as the shell streams it; the cases written to
// fixture (path) for the engine's think test to replay through its tracker. close and ends: the shell's reasoning
// sequences
int reasoning_check_utf8(const reasoning_check_ctx & c, const std::string & model, const std::vector<int32_t> & open,
                         const std::vector<int32_t> & close, const std::vector<std::vector<int32_t>> & ends,
                         uint32_t (*entry)(int32_t, const char *, size_t), const std::string & fixture);
