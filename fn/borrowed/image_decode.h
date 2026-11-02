#pragma once

// JPEG and PNG decoding on the CPU through stb_image (vendor/stb v2.30, borrowed; the decoder ggml-vfast 66e11c3's
// mtmd-helper.cpp uses), compiled with only its JPEG and PNG decoders (VIS1). The header is read first, so the caller
// checks the dimensions before any pixel is decoded. No other stb state is touched, so calls from several threads are
// independent.

#include <cstddef>
#include <cstdint>
#include <vector>

enum class fnfl_image_kind : int32_t { unknown, jpeg, png };

struct fnfl_image_info {
    fnfl_image_kind kind     = fnfl_image_kind::unknown;
    int32_t         width    = 0;
    int32_t         height   = 0;
    int32_t         channels = 0;  // as stored (1 to 4)
};

// The kind from the leading signature (JPEG FF D8 FF, PNG's 8-byte signature), or unknown.
fnfl_image_kind fnfl_image_sniff(const uint8_t * data, size_t n);

// The header only (stbi_info_from_memory): false when the bytes are not a JPEG or PNG header stb can read.
bool fnfl_image_probe(const uint8_t * data, size_t n, fnfl_image_info & info);

// The pixels as interleaved RGB8, rgb[(y w + x) 3 + c] (stbi_load_from_memory with 3 channels, as the reference). False
// on any decode error, with rgb cleared; max_pixels bounds w x h again after the decode (stb rechecks the header).
bool fnfl_image_decode_rgb8(const uint8_t * data, size_t n, int64_t max_pixels, std::vector<uint8_t> & rgb,
                            int32_t & width, int32_t & height);
