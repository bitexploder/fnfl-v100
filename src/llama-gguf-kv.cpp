#include "llama-gguf-kv.h"

#include "llama-impl.h"
#include "llama-vocab.h"

#include "gguf.h"

#include <memory>
#include <stdexcept>

llama_gguf_kv::llama_gguf_kv(gguf_context * metadata, llm_arch arch) : metadata(metadata), llm_kv(arch) {}

// the key's id, or -1 when absent; throws when it is present with another type
static int64_t find_typed(const gguf_context * ctx, const std::string & key, gguf_type type, bool required) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        if (required) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }
        return -1;
    }
    const gguf_type kt = gguf_get_kv_type(ctx, id);
    if (kt != type) {
        throw std::runtime_error(format("key %s has wrong type %s but expected type %s",
            key.c_str(), gguf_type_name(kt), gguf_type_name(type)));
    }
    return id;
}

bool llama_gguf_kv::get_key(enum llm_kv kid, std::string & result, bool required) {
    const int64_t id = find_typed(metadata, llm_kv(kid), GGUF_TYPE_STRING, required);
    if (id < 0) {
        return false;
    }
    result = gguf_get_val_str(metadata, id);
    return true;
}

bool llama_gguf_kv::get_key(enum llm_kv kid, uint32_t & result, bool required) {
    const int64_t id = find_typed(metadata, llm_kv(kid), GGUF_TYPE_UINT32, required);
    if (id < 0) {
        return false;
    }
    result = gguf_get_val_u32(metadata, id);
    return true;
}

bool llama_gguf_kv::get_key(enum llm_kv kid, bool & result, bool required) {
    const int64_t id = find_typed(metadata, llm_kv(kid), GGUF_TYPE_BOOL, required);
    if (id < 0) {
        return false;
    }
    result = gguf_get_val_bool(metadata, id);
    return true;
}

llama_vocab * llama_vocab_load_from_file(const char * path) {
    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx      = nullptr;

    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_from_file(path, params), gguf_free);
    if (!meta) {
        LLAMA_LOG_ERROR("%s: cannot read the GGUF file '%s'\n", __func__, path);
        return nullptr;
    }

    std::string arch_name;
    const int64_t arch_id = gguf_find_key(meta.get(), "general.architecture");
    if (arch_id >= 0 && gguf_get_kv_type(meta.get(), arch_id) == GGUF_TYPE_STRING) {
        arch_name = gguf_get_val_str(meta.get(), arch_id);
    }
    const llm_arch arch = llm_arch_from_string(arch_name);
    if (arch == LLM_ARCH_UNKNOWN) {
        LLAMA_LOG_ERROR("%s: unknown model architecture '%s'\n", __func__, arch_name.c_str());
        return nullptr;
    }

    auto vocab = std::make_unique<llama_vocab>();
    try {
        llama_gguf_kv kv(meta.get(), arch);
        vocab->load(kv);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: error loading the vocabulary of '%s': %s\n", __func__, path, e.what());
        return nullptr;
    }
    return vocab.release();
}

void llama_vocab_free(llama_vocab * vocab) {
    delete vocab;
}
