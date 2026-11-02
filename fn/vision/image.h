#pragma once

// VIS1: the vision module's CPU half. An image's bytes become the encoder's patch rows: decode (JPEG, PNG; bounded),
// resize to the pixel budget, normalize, and cut into 16 x 16 patches in merge-group order. The resize, the size rule
// and the normalization are adapted from ggml-vfast 66e11c3 (tools/mtmd/mtmd-image.cpp: img_tool's
// calc_size_preserved_ratio, resize with PAD_CEIL and resize_pillow's bicubic; clip-impl.h: clip_image_f32's from_u8
// and normalize), keeping their float operations so the normalized image is bitwise the reference's. Nothing here knows
// the text model, the split or the slots.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fn::vision {

constexpr int32_t patch       = 16;                 // pixels a patch side
constexpr int32_t merge       = 2;                  // 2 x 2 patches a token
constexpr int32_t align       = patch * merge;      // sides are multiples of 32
constexpr int32_t patch_vals  = 3 * patch * patch;  // 768: (c, ky, kx), the patch kernel's order
constexpr int32_t min_tokens  = 8;                  // the reference's set_limit_image_tokens(8, 4096) for Qwen3-VL
constexpr int32_t max_tokens  = 4096;
constexpr int64_t min_pixels  = int64_t(min_tokens) * align * align;  // 8,192
constexpr int64_t max_pixels  = int64_t(max_tokens) * align * align;  // 4,194,304
constexpr int32_t max_patches = max_tokens * merge * merge;           // 16,384

// The request limits for untrusted input (VIS1's decisions): 20 MB encoded an image, 50 megapixels decoded, checked
// from the header before any pixel is decoded, and 8 images a request. 46000 is the widest side the reference
// represents.
struct limits {
    int64_t max_bytes   = 20000000;
    int64_t max_decoded = 50000000;
    int32_t max_side    = 46000;
    int32_t max_images  = 8;
};

enum class status : int32_t {
    ok,
    empty,            // no bytes, or a zero width or height
    too_many_bytes,   // over limits::max_bytes
    not_supported,    // not a JPEG or PNG signature (WebP, GIF, HEIC, ...)
    bad_header,       // a JPEG or PNG signature whose header stb cannot read
    too_many_pixels,  // the header's w x h over limits::max_decoded
    side_too_large,   // a side over limits::max_side
    decode_failed,    // truncated or corrupt data
    too_many_images,  // over limits::max_images in one request
};

const char * status_text(status s);

// the request's image count against limits::max_images
status check_count(int32_t n_images, const limits & lim);

struct rgb8 {
    int32_t              w = 0, h = 0;
    std::vector<uint8_t> px;  // [h][w][3]
};

// The bounded decode. The header is checked first; then the pixels as RGB8 (alpha dropped, grey widened, 16-bit PNG
// narrowed, as stb's 3-channel load does for the reference).
status decode(const uint8_t * data, size_t n, const limits & lim, rgb8 & out);

struct size2 {
    int32_t w = 0, h = 0;
};

// The resized size (calc_size_preserved_ratio with align 32, min_pixels, max_pixels): each side rounded to a multiple
// of 32 (at least 32); over max_pixels, both sides floored after scaling by beta = sqrt(h w / max); under min_pixels,
// both ceiled after scaling up. {0, 0} for a non-positive side.
size2 target_size(int32_t w, int32_t h);

// The reference's resize with PAD_CEIL: scale = min(tw / w, th / h) (F32), the image resized (Pillow-compatible
// bicubic, a = -0.5, 22-bit fixed point) to min(ceil(w scale), tw) x min(ceil(h scale), th), centred on black at
// ((tw - nw) / 2, (th - nh) / 2). A copy when the sizes are equal.
void resize(const rgb8 & src, size2 t, rgb8 & dst);

// The normalized image: v = (u8 / 255 - 0.5) / 0.5 in F32, interleaved [h][w][3] (the reference's clip_image_f32).
void normalize(const rgb8 & img, std::vector<float> & hwc);

// The patch rows, [gw gh][768] F32, in merge-group order: patch (px, py) is row ((py / 2) (gw / 2) + px / 2) 4 +
// (py % 2) 2 + px % 2, its values hwc[((16 py + ky) w + 16 px + kx) 3 + c] at c 256 + ky 16 + kx. The 2 x 2 merge is
// then four consecutive rows. Requires w and h multiples of 32.
void cut_patches(const std::vector<float> & hwc, int32_t w, int32_t h, float * rows);

// One image prepared for the encoder: the resized size, the patch grid and the merged grid (tokens nx ny, row t at
// grid (t / nx, t % nx)), and the patch rows.
struct prepared {
    size2              src;             // decoded size
    size2              size;            // resized size
    int32_t            gw = 0, gh = 0;  // patches
    int32_t            nx = 0, ny = 0;  // tokens (gw / 2, gh / 2)
    std::vector<float> rows;            // [gw gh][768]
    // VIS1 Stage B: the image's content key (prefix reuse): a 64-bit hash of the encoded bytes and the resized size
    // (equal pixels in another encoding only miss a reuse)
    uint64_t           digest = 0;
};

// the digest of encoded bytes for a resized size (prepare's)
uint64_t content_digest(const uint8_t * data, size_t n, size2 size);

status prepare(const uint8_t * data, size_t n, const limits & lim, prepared & out);

}  // namespace fn::vision
