#pragma once

// The tensor directory of one GGUF file, read on the CPU through gguf.cpp (borrowed): each tensor's name, ggml type
// name, shape and byte range in the file. No tensor data is read. R1's loader maps the file and copies from these
// ranges.

#include <cstdint>
#include <string>
#include <vector>

struct fnfl_gguf_tensor {
    std::string name;
    std::string type_name; // ggml_type_name
    int32_t block_elems = 0;
    int32_t block_bytes = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    uint64_t offset = 0; // absolute byte offset of the tensor's data in the file
    uint64_t bytes = 0;
};

struct fnfl_gguf_dir {
    std::string path;
    uint64_t file_bytes = 0; // from the tensor ranges' end; the caller checks it against the file size
    std::vector<fnfl_gguf_tensor> tensors;
};

// Returns false and sets err on failure.
bool fnfl_gguf_read_dir(const std::string & path, fnfl_gguf_dir & dir, std::string & err);

// An integer key of one GGUF file (any integer width), or the first element of an integer array key; -1 when absent.
int64_t fnfl_gguf_read_int(const std::string & path, const std::string & key);

// Whether one GGUF file has a key (QA2's checks that the model sets no rope scaling, soft cap or ALiBi).
bool fnfl_gguf_has_key(const std::string & path, const std::string & key);

// A floating-point key (F32 or F64) of one GGUF file; false when it is absent or not a float.
bool fnfl_gguf_read_float(const std::string & path, const std::string & key, double & out);

// Every element of an integer array key of one GGUF file; false when it is absent or not integers.
bool fnfl_gguf_read_ints(const std::string & path, const std::string & key, std::vector<int64_t> & out);

// The paths of every shard of a split model, from its first shard's split.count (one path when it is not split).
bool fnfl_gguf_shard_paths(const std::string & first, std::vector<std::string> & paths, std::string & err);
