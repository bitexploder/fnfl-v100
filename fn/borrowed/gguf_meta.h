#pragma once

// Reads a model's metadata from its GGUF files on the CPU (gguf.cpp and llama.cpp's file-type names, borrowed). No tensor
// data is read.

#include <cstdint>
#include <string>

struct fnfl_gguf_meta {
    std::string arch;
    std::string name;
    std::string chat_template;
    std::string chat_template_tool_use;
    std::string format_name;

    int32_t n_shards = 0;
    int64_t n_tensors = 0;

    // the shape keys the glue checks (-1: absent)
    int64_t n_ctx_train = -1;
    int64_t n_embd = -1;
    int64_t n_layer = -1;
    int64_t n_head = -1;
    int64_t n_head_kv = -1;
    int64_t n_expert = -1;
    int64_t n_hc = -1;

    uint64_t n_params = 0;
    uint64_t size = 0;
};

// path: the first shard; the others are found from its split.count. Returns false and sets err on failure
bool fnfl_gguf_read_meta(const std::string & path, fnfl_gguf_meta & meta, std::string & err);
