#include "fn/borrowed/gguf_meta.h"

#include "gguf.h"
#include "llama.h"

#include <memory>
#include <vector>

namespace {

struct gguf_deleter {
    void operator()(gguf_context * ctx) const { gguf_free(ctx); }
};
using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;

gguf_ptr open_meta(const std::string & path) {
    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = nullptr;
    return gguf_ptr(gguf_init_from_file(path.c_str(), params));
}

std::string get_str(const gguf_context * ctx, const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return std::string();
    }
    return gguf_get_val_str(ctx, id);
}

// an unsigned or signed integer key of any width, -1 when absent or not an integer
int64_t get_int(const gguf_context * ctx, const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        return -1;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:
            return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_INT8:
            return gguf_get_val_i8(ctx, id);
        case GGUF_TYPE_UINT16:
            return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16:
            return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32:
            return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:
            return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64:
            return (int64_t)gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64:
            return gguf_get_val_i64(ctx, id);
        default:
            return -1;
    }
}

// the elements and bytes of every tensor in one shard
void add_tensors(const gguf_context * ctx, fnfl_gguf_meta & meta) {
    const int64_t n = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n; ++i) {
        const ggml_type type = gguf_get_tensor_type(ctx, i);
        const size_t bytes = gguf_get_tensor_size(ctx, i);
        meta.size += bytes;
        meta.n_params += (uint64_t)(bytes / ggml_type_size(type)) * (uint64_t)ggml_blck_size(type);
    }
    meta.n_tensors += n;
}

} // namespace

bool fnfl_gguf_read_meta(const std::string & path, fnfl_gguf_meta & meta, std::string & err) {
    meta = fnfl_gguf_meta();

    gguf_ptr first = open_meta(path);
    if (!first) {
        err = "cannot read the GGUF file '" + path + "'";
        return false;
    }
    const gguf_context * ctx = first.get();

    meta.arch = get_str(ctx, "general.architecture");
    meta.name = get_str(ctx, "general.name");

    meta.chat_template = get_str(ctx, "tokenizer.chat_template");
    meta.chat_template_tool_use = get_str(ctx, "tokenizer.chat_template.tool_use");

    const std::string a = meta.arch + ".";
    meta.n_ctx_train = get_int(ctx, a + "context_length");
    meta.n_embd = get_int(ctx, a + "embedding_length");
    meta.n_layer = get_int(ctx, a + "block_count");
    meta.n_head = get_int(ctx, a + "attention.head_count");
    meta.n_head_kv = get_int(ctx, a + "attention.head_count_kv");
    meta.n_expert = get_int(ctx, a + "expert_count");
    meta.n_hc = get_int(ctx, a + "hyper_connection.count");

    const int64_t ftype = get_int(ctx, "general.file_type");
    meta.format_name = ftype >= 0 ? llama_ftype_name((llama_ftype)ftype) : "unknown";

    add_tensors(ctx, meta);

    const int64_t n_split = get_int(ctx, "split.count");
    meta.n_shards = n_split > 1 ? (int32_t)n_split : 1;
    if (meta.n_shards > 1) {
        std::vector<char> prefix(path.size() + 1);
        if (llama_split_prefix(prefix.data(), prefix.size(), path.c_str(), 0, meta.n_shards) <= 0) {
            err = "'" + path + "' is not the first shard of a split model";
            return false;
        }
        std::vector<char> shard(path.size() + 32);
        for (int32_t i = 1; i < meta.n_shards; ++i) {
            llama_split_path(shard.data(), shard.size(), prefix.data(), i, meta.n_shards);
            gguf_ptr other = open_meta(shard.data());
            if (!other) {
                err = std::string("cannot read the GGUF shard '") + shard.data() + "'";
                return false;
            }
            add_tensors(other.get(), meta);
        }
    }
    return true;
}
