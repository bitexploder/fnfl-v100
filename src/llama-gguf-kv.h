#pragma once

#include "llama-arch.h"

#include <cstdint>
#include <string>

struct gguf_context;

// FNFL (FL2): the GGUF key reader the tokenizer loads with, in place of llama_model_loader. Keys are read with the same exact
// type check as the loader's GGUFMeta; there are no key overrides.
struct llama_gguf_kv {
    llama_gguf_kv(gguf_context * metadata, llm_arch arch);

    gguf_context * metadata;
    LLM_KV         llm_kv;

    bool get_key(enum llm_kv kid, std::string & result, bool required = true);
    bool get_key(enum llm_kv kid, uint32_t    & result, bool required = true);
    bool get_key(enum llm_kv kid, bool        & result, bool required = true);
};
