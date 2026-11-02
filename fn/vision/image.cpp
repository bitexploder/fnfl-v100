#include "fn/vision/image.h"

#include "fn/borrowed/image_decode.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace fn::vision {

const char * status_text(status s) {
    switch (s) {
        case status::ok:
            return "ok";
        case status::empty:
            return "empty image";
        case status::too_many_bytes:
            return "image over the size limit";
        case status::not_supported:
            return "unsupported image format (JPEG and PNG only)";
        case status::bad_header:
            return "unreadable image header";
        case status::too_many_pixels:
            return "image over the pixel limit";
        case status::side_too_large:
            return "image side over the limit";
        case status::decode_failed:
            return "image data is corrupt or truncated";
        case status::too_many_images:
            return "too many images in the request";
    }
    return "unknown image status";
}

status check_count(int32_t n_images, const limits & lim) {
    return n_images > lim.max_images ? status::too_many_images : status::ok;
}

status decode(const uint8_t * data, size_t n, const limits & lim, rgb8 & out) {
    out = rgb8();
    if (data == nullptr || n == 0) {
        return status::empty;
    }
    if (int64_t(n) > lim.max_bytes) {
        return status::too_many_bytes;
    }
    if (fnfl_image_sniff(data, n) == fnfl_image_kind::unknown) {
        return status::not_supported;
    }
    fnfl_image_info info;
    if (!fnfl_image_probe(data, n, info)) {
        return status::bad_header;
    }
    if (info.width <= 0 || info.height <= 0) {
        return status::empty;
    }
    if (info.width > lim.max_side || info.height > lim.max_side) {
        return status::side_too_large;
    }
    if (int64_t(info.width) * int64_t(info.height) > lim.max_decoded) {
        return status::too_many_pixels;
    }
    int32_t w = 0, h = 0;
    if (!fnfl_image_decode_rgb8(data, n, lim.max_decoded, out.px, w, h)) {
        out = rgb8();
        return status::decode_failed;
    }
    // stb decodes the frame the header named; a different size means the header lied
    if (w != info.width || h != info.height) {
        out = rgb8();
        return status::decode_failed;
    }
    out.w = w;
    out.h = h;
    return status::ok;
}

size2 target_size(int32_t width, int32_t height) {
    if (width <= 0 || height <= 0) {
        return {};
    }
    const int32_t f       = align;
    const auto    round_f = [f](float x) {
        return static_cast<int32_t>(std::round(x / static_cast<float>(f))) * f;
    };
    const auto ceil_f = [f](float x) {
        return static_cast<int32_t>(std::ceil(x / static_cast<float>(f))) * f;
    };
    const auto floor_f = [f](float x) {
        return static_cast<int32_t>(std::floor(x / static_cast<float>(f))) * f;
    };
    int32_t w_bar = std::max(f, round_f(static_cast<float>(width)));
    int32_t h_bar = std::max(f, round_f(static_cast<float>(height)));
    if (int64_t(h_bar) * w_bar > max_pixels) {
        const float beta = std::sqrt(static_cast<float>(height) * width / static_cast<int32_t>(max_pixels));
        h_bar            = std::max(f, floor_f(height / beta));
        w_bar            = std::max(f, floor_f(width / beta));
    } else if (int64_t(h_bar) * w_bar < min_pixels) {
        const float beta =
            std::sqrt(static_cast<float>(static_cast<int32_t>(min_pixels)) / (static_cast<float>(height) * width));
        h_bar = ceil_f(height * beta);
        w_bar = ceil_f(width * beta);
    }
    return { w_bar, h_bar };
}

namespace {

constexpr int32_t precision_bits = 32 - 8 - 2;

double bicubic(double x) {
    constexpr double a = -0.5;  // Pillow's bicubic
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    }
    if (x < 2.0) {
        return (((x - 5) * x + 8) * x - 4) * a;
    }
    return 0.0;
}

uint8_t clip8(int32_t v) {
    return static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
}

// one dimension's taps: bounds[2 o] the first input, bounds[2 o + 1] the count, w[o ksize + i] the fixed-point weights
int32_t taps(int32_t in, int32_t out, std::vector<int32_t> & bounds, std::vector<int32_t> & w) {
    const double        scale       = static_cast<double>(in) / out;
    const double        filterscale = scale < 1.0 ? 1.0 : scale;
    const double        support     = 2.0 * filterscale;
    const int32_t       ksize       = static_cast<int32_t>(std::ceil(support)) * 2 + 1;
    std::vector<double> pre(size_t(out) * ksize);
    bounds.assign(size_t(out) * 2, 0);
    for (int32_t xx = 0; xx < out; ++xx) {
        const double center = (xx + 0.5) * scale;
        const double ss     = 1.0 / filterscale;
        double       ww     = 0.0;
        int32_t      xmin   = static_cast<int32_t>(center - support + 0.5);
        if (xmin < 0) {
            xmin = 0;
        }
        int32_t xmax = static_cast<int32_t>(center + support + 0.5);
        if (xmax > in) {
            xmax = in;
        }
        xmax -= xmin;
        int32_t x = 0;
        for (; x < xmax; ++x) {
            const double v              = bicubic((x + xmin - center + 0.5) * ss);
            pre[size_t(xx) * ksize + x] = v;
            ww += v;
        }
        for (x = 0; x < xmax; ++x) {
            if (ww != 0.0) {
                pre[size_t(xx) * ksize + x] /= ww;
            }
        }
        for (; x < ksize; ++x) {
            pre[size_t(xx) * ksize + x] = 0;
        }
        bounds[size_t(xx) * 2]     = xmin;
        bounds[size_t(xx) * 2 + 1] = xmax;
    }
    w.resize(pre.size());
    const double fxp = std::ldexp(1.0, precision_bits);
    for (size_t i = 0; i < pre.size(); ++i) {
        // Pillow adds +/- 0.5 then truncates toward zero
        w[i] = static_cast<int32_t>(pre[i] * fxp + (pre[i] < 0 ? -0.5 : 0.5));
    }
    return ksize;
}

std::vector<uint8_t> horizontal(const uint8_t *              src,
                                int32_t                      in_w,
                                int32_t                      h,
                                int32_t                      out_w,
                                int32_t                      ksize,
                                const std::vector<int32_t> & bounds,
                                const std::vector<int32_t> & w) {
    std::vector<uint8_t> out(size_t(out_w) * h * 3);
    for (int32_t y = 0; y < h; ++y) {
        const uint8_t * row = src + size_t(y) * in_w * 3;
        uint8_t *       dst = out.data() + size_t(y) * out_w * 3;
        for (int32_t xx = 0; xx < out_w; ++xx) {
            const int32_t   xmin = bounds[size_t(xx) * 2], xcnt = bounds[size_t(xx) * 2 + 1];
            const int32_t * k  = &w[size_t(xx) * ksize];
            const uint8_t * p  = row + size_t(xmin) * 3;
            int32_t         s0 = 1 << (precision_bits - 1), s1 = s0, s2 = s0;
            for (int32_t x = 0; x < xcnt; ++x) {
                s0 += p[0] * k[x];
                s1 += p[1] * k[x];
                s2 += p[2] * k[x];
                p += 3;
            }
            dst[xx * 3 + 0] = clip8(s0 >> precision_bits);
            dst[xx * 3 + 1] = clip8(s1 >> precision_bits);
            dst[xx * 3 + 2] = clip8(s2 >> precision_bits);
        }
    }
    return out;
}

std::vector<uint8_t> vertical(const uint8_t *              src,
                              int32_t                      w_px,
                              int32_t                      out_h,
                              int32_t                      ksize,
                              const std::vector<int32_t> & bounds,
                              const std::vector<int32_t> & w) {
    const size_t         n = size_t(w_px) * 3;
    std::vector<uint8_t> out(n * out_h);
    std::vector<int32_t> acc(n);
    for (int32_t yy = 0; yy < out_h; ++yy) {
        const int32_t   ymin = bounds[size_t(yy) * 2], ycnt = bounds[size_t(yy) * 2 + 1];
        const int32_t * k = &w[size_t(yy) * ksize];
        std::fill(acc.begin(), acc.end(), 1 << (precision_bits - 1));
        for (int32_t y = 0; y < ycnt; ++y) {
            const uint8_t * row = src + size_t(ymin + y) * n;
            const int32_t   wy  = k[y];
            for (size_t i = 0; i < n; ++i) {
                acc[i] += row[i] * wy;
            }
        }
        uint8_t * dst = out.data() + size_t(yy) * n;
        for (size_t i = 0; i < n; ++i) {
            dst[i] = clip8(acc[i] >> precision_bits);
        }
    }
    return out;
}

// resize_pillow's bicubic: the horizontal pass, then the vertical, each only when that side changes
void resample(const rgb8 & src, int32_t tw, int32_t th, rgb8 & dst) {
    const bool           need_h = tw != src.w, need_v = th != src.h;
    std::vector<int32_t> bh, wh, bv, wv;
    const int32_t        kh = need_h ? taps(src.w, tw, bh, wh) : 0;
    const int32_t        kv = need_v ? taps(src.h, th, bv, wv) : 0;
    dst.w                   = tw;
    dst.h                   = th;
    if (need_h && need_v) {
        const std::vector<uint8_t> tmp = horizontal(src.px.data(), src.w, src.h, tw, kh, bh, wh);
        dst.px                         = vertical(tmp.data(), tw, th, kv, bv, wv);
    } else if (need_h) {
        dst.px = horizontal(src.px.data(), src.w, src.h, tw, kh, bh, wh);
    } else if (need_v) {
        dst.px = vertical(src.px.data(), src.w, th, kv, bv, wv);
    } else {
        dst.px = src.px;
    }
}

}  // namespace

void resize(const rgb8 & src, size2 t, rgb8 & dst) {
    dst = rgb8();
    if (t.w == src.w && t.h == src.h) {
        dst = src;
        return;
    }
    const float   scale_w = static_cast<float>(t.w) / src.w;
    const float   scale_h = static_cast<float>(t.h) / src.h;
    const float   scale   = std::min(scale_w, scale_h);
    const int32_t nw      = std::min(static_cast<int32_t>(std::ceil(src.w * scale)), t.w);
    const int32_t nh      = std::min(static_cast<int32_t>(std::ceil(src.h * scale)), t.h);
    rgb8          mid;
    resample(src, nw, nh, mid);
    dst.w = t.w;
    dst.h = t.h;
    dst.px.assign(size_t(t.w) * t.h * 3, 0);
    const int32_t ox = (t.w - nw) / 2, oy = (t.h - nh) / 2;
    for (int32_t y = 0; y < nh; ++y) {
        const int32_t dy = y + oy;
        if (dy < 0 || dy >= t.h) {
            continue;
        }
        for (int32_t x = 0; x < nw; ++x) {
            const int32_t dx = x + ox;
            if (dx < 0 || dx >= t.w) {
                continue;
            }
            const size_t s = (size_t(y) * nw + x) * 3, d = (size_t(dy) * t.w + dx) * 3;
            dst.px[d]     = mid.px[s];
            dst.px[d + 1] = mid.px[s + 1];
            dst.px[d + 2] = mid.px[s + 2];
        }
    }
}

void normalize(const rgb8 & img, std::vector<float> & hwc) {
    const float mean = 0.5f, sd = 0.5f;  // the file's image_mean and image_std (the profile checks them)
    hwc.resize(img.px.size());
    for (size_t i = 0; i < img.px.size(); ++i) {
        hwc[i] = (static_cast<float>(img.px[i]) / 255.0f - mean) / sd;
    }
}

void cut_patches(const std::vector<float> & hwc, int32_t w, int32_t h, float * rows) {
    const int32_t gw = w / patch, gh = h / patch;
    for (int32_t py = 0; py < gh; ++py) {
        for (int32_t px = 0; px < gw; ++px) {
            const int64_t r   = (int64_t(py / 2) * (gw / 2) + px / 2) * 4 + (py % 2) * 2 + px % 2;
            float *       dst = rows + r * patch_vals;
            for (int32_t c = 0; c < 3; ++c) {
                for (int32_t ky = 0; ky < patch; ++ky) {
                    const float * src = hwc.data() + ((size_t(py) * patch + ky) * w + size_t(px) * patch) * 3 + c;
                    for (int32_t kx = 0; kx < patch; ++kx) {
                        dst[c * 256 + ky * 16 + kx] = src[size_t(kx) * 3];
                    }
                }
            }
        }
    }
}

uint64_t content_digest(const uint8_t * data, size_t n, size2 size) {
    // eight bytes at a time, each word mixed in by a multiply and a shift (the tail zero-padded), then the length
    // and the size
    const uint64_t k   = 0x9e3779b97f4a7c15ull;
    uint64_t       h   = 0x51ed270b27f1a3c5ull;
    const auto     mix = [&](uint64_t w) {
        h = (h ^ w) * k;
        h ^= h >> 29;
    };
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        std::memcpy(&w, data + i, 8);
        mix(w);
    }
    uint64_t tail = 0;
    std::memcpy(&tail, data + i, n - i);
    mix(tail);
    mix(uint64_t(n));
    mix((uint64_t(uint32_t(size.w)) << 32) | uint32_t(size.h));
    return h;
}

status prepare(const uint8_t * data, size_t n, const limits & lim, prepared & out) {
    out = prepared();
    rgb8         img;
    const status s = decode(data, n, lim, img);
    if (s != status::ok) {
        return s;
    }
    out.src  = { img.w, img.h };
    out.size = target_size(img.w, img.h);
    if (out.size.w <= 0 || out.size.h <= 0) {
        return status::empty;
    }
    rgb8 sized;
    resize(img, out.size, sized);
    img = rgb8();
    std::vector<float> hwc;
    normalize(sized, hwc);
    out.gw = out.size.w / patch;
    out.gh = out.size.h / patch;
    out.nx = out.gw / merge;
    out.ny = out.gh / merge;
    out.rows.resize(size_t(out.gw) * out.gh * patch_vals);
    cut_patches(hwc, out.size.w, out.size.h, out.rows.data());
    out.digest = content_digest(data, n, out.size);
    return status::ok;
}

}  // namespace fn::vision
