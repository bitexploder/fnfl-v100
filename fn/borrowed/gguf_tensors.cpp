#include "fn/borrowed/gguf_tensors.h"

#include "gguf.h"
#include "llama.h"

#include <memory>

namespace {

struct gguf_deleter {
    void operator()(gguf_context * ctx) const { gguf_free(ctx); }
};
using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;

struct ggml_ctx_deleter {
    void operator()(ggml_context * ctx) const { ggml_free(ctx); }
};
using ggml_ctx_ptr = std::unique_ptr<ggml_context, ggml_ctx_deleter>;

// an integer key of any width, -1 when it is not an integer
int64_t scalar_int(const gguf_context * ctx, int64_t id) {
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8: return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_INT8: return gguf_get_val_i8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16: return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32: return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return (int64_t)gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64: return gguf_get_val_i64(ctx, id);
        default: return -1;
    }
}

// element 0 of an integer array key, -1 when it is empty or not integers
int64_t array_int0(const gguf_context * ctx, int64_t id) {
    if (gguf_get_arr_n(ctx, id) == 0) {
        return -1;
    }
    const void * p = gguf_get_arr_data(ctx, id);
    switch (gguf_get_arr_type(ctx, id)) {
        case GGUF_TYPE_UINT8: return *static_cast<const uint8_t *>(p);
        case GGUF_TYPE_INT8: return *static_cast<const int8_t *>(p);
        case GGUF_TYPE_UINT16: return *static_cast<const uint16_t *>(p);
        case GGUF_TYPE_INT16: return *static_cast<const int16_t *>(p);
        case GGUF_TYPE_UINT32: return *static_cast<const uint32_t *>(p);
        case GGUF_TYPE_INT32: return *static_cast<const int32_t *>(p);
        case GGUF_TYPE_UINT64: return (int64_t)*static_cast<const uint64_t *>(p);
        case GGUF_TYPE_INT64: return *static_cast<const int64_t *>(p);
        default: return -1;
    }
}

} // namespace

bool fnfl_gguf_read_dir(const std::string & path, fnfl_gguf_dir & dir, std::string & err) {
    dir = fnfl_gguf_dir();
    dir.path = path;
    ggml_context * meta = nullptr;
    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = &meta; // the tensors' shapes, without their data
    gguf_ptr ctx(gguf_init_from_file(path.c_str(), params));
    ggml_ctx_ptr meta_owner(meta);
    if (!ctx || meta == nullptr) {
        err = "cannot read the GGUF file '" + path + "'";
        return false;
    }
    const size_t data_offset = gguf_get_data_offset(ctx.get());
    const int64_t n = gguf_get_n_tensors(ctx.get());
    dir.tensors.reserve((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(ctx.get(), i);
        const ggml_tensor * t = ggml_get_tensor(meta, name);
        if (t == nullptr) {
            err = std::string("GGUF tensor '") + name + "' has no shape in '" + path + "'";
            return false;
        }
        fnfl_gguf_tensor d;
        d.name = name;
        d.type_name = ggml_type_name(t->type);
        d.block_elems = (int32_t)ggml_blck_size(t->type);
        d.block_bytes = (int32_t)ggml_type_size(t->type);
        for (int32_t k = 0; k < 4; ++k) {
            d.ne[k] = t->ne[k];
        }
        d.offset = data_offset + gguf_get_tensor_offset(ctx.get(), i);
        d.bytes = gguf_get_tensor_size(ctx.get(), i);
        if (d.offset + d.bytes > dir.file_bytes) {
            dir.file_bytes = d.offset + d.bytes;
        }
        dir.tensors.push_back(d);
    }
    return true;
}

int64_t fnfl_gguf_read_int(const std::string & path, const std::string & key) {
    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = nullptr;
    gguf_ptr ctx(gguf_init_from_file(path.c_str(), params));
    if (!ctx) {
        return -1;
    }
    const int64_t id = gguf_find_key(ctx.get(), key.c_str());
    if (id < 0) {
        return -1;
    }
    return gguf_get_kv_type(ctx.get(), id) == GGUF_TYPE_ARRAY ? array_int0(ctx.get(), id) : scalar_int(ctx.get(), id);
}

namespace {

// the key's id in an opened file, -1 when the file or the key is missing
struct key_ref {
    gguf_ptr ctx;
    int64_t id = -1;
};

key_ref find_key(const std::string & path, const std::string & key) {
    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = nullptr;
    key_ref r;
    r.ctx.reset(gguf_init_from_file(path.c_str(), params));
    if (r.ctx) {
        r.id = gguf_find_key(r.ctx.get(), key.c_str());
    }
    return r;
}

} // namespace

bool fnfl_gguf_has_key(const std::string & path, const std::string & key) {
    return find_key(path, key).id >= 0;
}

bool fnfl_gguf_read_float(const std::string & path, const std::string & key, double & out) {
    const key_ref r = find_key(path, key);
    if (r.id < 0) {
        return false;
    }
    switch (gguf_get_kv_type(r.ctx.get(), r.id)) {
        case GGUF_TYPE_FLOAT32:
            out = gguf_get_val_f32(r.ctx.get(), r.id);
            return true;
        case GGUF_TYPE_FLOAT64:
            out = gguf_get_val_f64(r.ctx.get(), r.id);
            return true;
        default:
            return false;
    }
}

bool fnfl_gguf_read_ints(const std::string & path, const std::string & key, std::vector<int64_t> & out) {
    out.clear();
    const key_ref r = find_key(path, key);
    if (r.id < 0 || gguf_get_kv_type(r.ctx.get(), r.id) != GGUF_TYPE_ARRAY) {
        return false;
    }
    const size_t n = gguf_get_arr_n(r.ctx.get(), r.id);
    const void * p = gguf_get_arr_data(r.ctx.get(), r.id);
    for (size_t i = 0; i < n; ++i) {
        switch (gguf_get_arr_type(r.ctx.get(), r.id)) {
            case GGUF_TYPE_INT32:
                out.push_back(static_cast<const int32_t *>(p)[i]);
                break;
            case GGUF_TYPE_UINT32:
                out.push_back(static_cast<const uint32_t *>(p)[i]);
                break;
            case GGUF_TYPE_INT64:
                out.push_back(static_cast<const int64_t *>(p)[i]);
                break;
            case GGUF_TYPE_UINT64:
                out.push_back(static_cast<int64_t>(static_cast<const uint64_t *>(p)[i]));
                break;
            default:
                return false;
        }
    }
    return true;
}

bool fnfl_gguf_shard_paths(const std::string & first, std::vector<std::string> & paths, std::string & err) {
    paths.clear();
    const int64_t n_split = fnfl_gguf_read_int(first, "split.count");
    if (n_split <= 1) {
        paths.push_back(first);
        return true;
    }
    std::vector<char> prefix(first.size() + 1);
    if (llama_split_prefix(prefix.data(), prefix.size(), first.c_str(), 0, (int)n_split) <= 0) {
        err = "'" + first + "' is not the first shard of a split model";
        return false;
    }
    std::vector<char> shard(first.size() + 32);
    for (int32_t i = 0; i < (int32_t)n_split; ++i) {
        llama_split_path(shard.data(), shard.size(), prefix.data(), i, (int)n_split);
        paths.emplace_back(shard.data());
    }
    return true;
}
