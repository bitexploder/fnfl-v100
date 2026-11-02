#include "fn/borrowed/image_decode.h"

#include <climits>
#include <cstring>

// stb_image, its JPEG and PNG decoders only, static to this file. 46000 is the largest side the reference represents
// (clip_image_size::area's assert); stb refuses a larger header itself.
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_MAX_DIMENSIONS 46000
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-function"
#    pragma GCC diagnostic ignored "-Wsign-compare"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#    pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#    pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#    pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include "stb/stb_image.h"
#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif

fnfl_image_kind fnfl_image_sniff(const uint8_t * data, size_t n) {
    static const uint8_t png_sig[8] = { 0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a };
    if (data == nullptr) {
        return fnfl_image_kind::unknown;
    }
    if (n >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff) {
        return fnfl_image_kind::jpeg;
    }
    if (n >= 8 && std::memcmp(data, png_sig, 8) == 0) {
        return fnfl_image_kind::png;
    }
    return fnfl_image_kind::unknown;
}

bool fnfl_image_probe(const uint8_t * data, size_t n, fnfl_image_info & info) {
    info = fnfl_image_info();
    info.kind = fnfl_image_sniff(data, n);
    if (info.kind == fnfl_image_kind::unknown || n > size_t(INT_MAX)) {
        return false;
    }
    int w = 0, h = 0, c = 0;
    if (stbi_info_from_memory(data, static_cast<int>(n), &w, &h, &c) == 0) {
        return false;
    }
    info.width    = w;
    info.height   = h;
    info.channels = c;
    return true;
}

bool fnfl_image_decode_rgb8(const uint8_t * data, size_t n, int64_t max_pixels, std::vector<uint8_t> & rgb,
                            int32_t & width, int32_t & height) {
    rgb.clear();
    width  = 0;
    height = 0;
    if (fnfl_image_sniff(data, n) == fnfl_image_kind::unknown || n > size_t(INT_MAX)) {
        return false;
    }
    int       w = 0, h = 0, c = 0;
    stbi_uc * px = stbi_load_from_memory(data, static_cast<int>(n), &w, &h, &c, 3);
    if (px == nullptr) {
        return false;
    }
    const int64_t pixels = int64_t(w) * int64_t(h);
    if (w <= 0 || h <= 0 || pixels > max_pixels) {
        stbi_image_free(px);
        return false;
    }
    rgb.assign(px, px + pixels * 3);
    stbi_image_free(px);
    width  = w;
    height = h;
    return true;
}
