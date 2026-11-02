#include "fn/vision/vision_test.h"
#include "fn/vision/image.h"
#include "fn/vision/vision_ref.h"
#include "fn/vision/weights.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>

namespace fn::vision::test {

bool read_dump(const std::string & path, dump & d) {
    d = dump();
    std::ifstream f(path, std::ios::binary);
    char          magic[8];
    if (!f.read(magic, 8) || std::memcmp(magic, "VIS1DMP", 8) != 0 || !f.read(reinterpret_cast<char *>(&d.type), 4) ||
        !f.read(reinterpret_cast<char *>(&d.ndim), 4) || !f.read(reinterpret_cast<char *>(d.dims), 32) || d.type != 0) {
        return false;
    }
    int64_t n = 1;
    for (int32_t i = 0; i < 4; ++i) {
        if (d.dims[i] < 1 || d.dims[i] > (int64_t(1) << 31)) {
            return false;
        }
        n *= d.dims[i];
    }
    d.f32.resize(size_t(n));
    return bool(f.read(reinterpret_cast<char *>(d.f32.data()), n * 4));
}

namespace {

template <typename R> row_err compare_any(const float * a, const R * ref, int32_t rows, int32_t n) {
    row_err e;
    double  gmax = 0, gerr = 0;
    for (int32_t r = 0; r < rows; ++r) {
        double num = 0, den = 0;
        for (int32_t c = 0; c < n; ++c) {
            const double x = a[size_t(r) * n + c], y = double(ref[size_t(r) * n + c]);
            num += (x - y) * (x - y);
            den += y * y;
            gmax = std::max(gmax, std::fabs(y));
            gerr = std::max(gerr, std::fabs(x - y));
        }
        const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
        e.mean_rel += rel;
        if (rel > e.max_rel || e.worst_row < 0) {
            e.max_rel   = rel;
            e.worst_row = r;
        }
    }
    e.mean_rel /= std::max(1, rows);
    e.max_abs_over_max = gmax > 0 ? gerr / gmax : gerr;
    return e;
}

std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::string base_of(const std::string & path) {
    const size_t s = path.find_last_of('/');
    std::string  b = s == std::string::npos ? path : path.substr(s + 1);
    const size_t d = b.find_last_of('.');
    return d == std::string::npos ? b : b.substr(0, d);
}

uint32_t crc32(const uint8_t * p, size_t n) {
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int32_t k = 0; k < 8; ++k) {
            c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
        }
    }
    return ~c;
}

void be32(std::vector<uint8_t> & v, uint32_t x) {
    for (int32_t i = 3; i >= 0; --i) {
        v.push_back(uint8_t(x >> (8 * i)));
    }
}

// a PNG whose IHDR says w x h (valid CRC) followed by an IDAT of `idat` raw bytes and IEND
std::vector<uint8_t> png_header(uint32_t w, uint32_t h, size_t idat) {
    std::vector<uint8_t> v     = { 0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a };
    const auto           chunk = [&](const char * type, const std::vector<uint8_t> & data) {
        be32(v, uint32_t(data.size()));
        std::vector<uint8_t> td(type, type + 4);
        td.insert(td.end(), data.begin(), data.end());
        v.insert(v.end(), td.begin(), td.end());
        be32(v, crc32(td.data(), td.size()));
    };
    std::vector<uint8_t> ihdr;
    be32(ihdr, w);
    be32(ihdr, h);
    ihdr.insert(ihdr.end(), { 8, 2, 0, 0, 0 });
    chunk("IHDR", ihdr);
    chunk("IDAT", std::vector<uint8_t>(idat, 0x78));
    chunk("IEND", {});
    return v;
}

// a copy of a JPEG with its SOF0/SOF2 frame size set to w x h
std::vector<uint8_t> jpeg_resized_header(std::vector<uint8_t> j, uint16_t w, uint16_t h) {
    for (size_t i = 2; i + 9 < j.size(); ++i) {
        if (j[i] == 0xff && (j[i + 1] == 0xc0 || j[i + 1] == 0xc2)) {
            j[i + 5] = uint8_t(h >> 8);
            j[i + 6] = uint8_t(h);
            j[i + 7] = uint8_t(w >> 8);
            j[i + 8] = uint8_t(w);
            break;
        }
    }
    return j;
}

}  // namespace

row_err compare_rows(const float * a, const double * ref, int32_t rows, int32_t n) {
    return compare_any(a, ref, rows, n);
}

row_err compare_rows(const float * a, const float * ref, int32_t rows, int32_t n) {
    return compare_any(a, ref, rows, n);
}

int run_prep(const std::vector<std::string> & images, const std::string & ref_dir) {
    int fails = 0;
    for (const std::string & path : images) {
        const std::vector<uint8_t> bytes = read_file(path);
        rgb8                       img;
        const status               s = decode(bytes.data(), bytes.size(), limits(), img);
        dump                       d;
        if (s != status::ok || !read_dump(ref_dir + "/" + base_of(path) + ".prep.bin", d)) {
            std::printf("prep %-24s FAIL (%s, or no reference dump)\n", base_of(path).c_str(), status_text(s));
            ++fails;
            continue;
        }
        const size2 t = target_size(img.w, img.h);
        rgb8        sized;
        resize(img, t, sized);
        std::vector<float> hwc;
        normalize(sized, hwc);
        size_t diff = 0;
        if (d.dims[1] != t.w || d.dims[2] != t.h || d.f32.size() != hwc.size()) {
            diff = hwc.size() + 1;
        } else {
            for (size_t i = 0; i < hwc.size(); ++i) {
                diff += std::memcmp(&hwc[i], &d.f32[i], 4) != 0;
            }
        }
        // the patch cut is a permutation: each row value is its pixel's
        prepared p;
        size_t   cut_bad = 0;
        if (prepare(bytes.data(), bytes.size(), limits(), p) != status::ok || p.size.w != t.w || p.size.h != t.h) {
            cut_bad = 1;
        } else {
            for (int32_t py = 0; py < p.gh; ++py) {
                for (int32_t px = 0; px < p.gw; ++px) {
                    const int64_t r = (int64_t(py / 2) * (p.gw / 2) + px / 2) * 4 + (py % 2) * 2 + px % 2;
                    for (int32_t k = 0; k < patch_vals; k += 37) {
                        const int32_t c = k / 256, ky = (k / 16) % 16, kx = k % 16;
                        const float   v = hwc[((size_t(py) * 16 + ky) * t.w + size_t(px) * 16 + kx) * 3 + c];
                        cut_bad += std::memcmp(&v, &p.rows[size_t(r) * patch_vals + k], 4) != 0;
                    }
                }
            }
        }
        const bool ok = diff == 0 && cut_bad == 0;
        std::printf(
            "prep %-24s %5dx%-5d -> %5dx%-5d tokens %4d: %zu of %zu values differ from the reference, cut %s: %s\n",
            base_of(path).c_str(), img.w, img.h, t.w, t.h, (t.w / 32) * (t.h / 32), diff, hwc.size(),
            cut_bad ? "WRONG" : "ok", ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    std::printf("prep: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

int run_limits(const std::string & images_dir) {
    struct cas {
        std::string          name;
        std::vector<uint8_t> bytes;
        limits               lim;
        std::vector<status>  want;
    };

    std::vector<cas> c;
    const limits     def;
    c.push_back({ "empty buffer", {}, def, { status::empty } });
    {
        std::vector<uint8_t> big = png_header(64, 64, 16);
        big.resize(size_t(def.max_bytes) + 1, 0);
        c.push_back({ "21 MB over 20 MB", big, def, { status::too_many_bytes } });
    }
    c.push_back({
        "GIF signature", { 'G', 'I', 'F', '8', '9', 'a', 1, 0, 1, 0, 0, 0, 0 },
         def, { status::not_supported }
    });
    c.push_back({
        "WebP signature",
        { 'R', 'I', 'F', 'F', 8, 0, 0, 0, 'W', 'E', 'B', 'P', 'V', 'P', '8', ' ' },
        def,
        { status::not_supported }
    });
    c.push_back({
        "HEIC (ftyp)",
        { 0, 0, 0, 24, 'f', 't', 'y', 'p', 'h', 'e', 'i', 'c', 0, 0, 0, 0 },
        def,
        { status::not_supported }
    });
    c.push_back({
        "BMP signature", { 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0 },
         def, { status::not_supported }
    });
    {
        std::mt19937         g(7);
        std::vector<uint8_t> r(4096);
        for (auto & b : r) {
            b = uint8_t(g());
        }
        r[0] = 0x00;
        c.push_back({ "random bytes", r, def, { status::not_supported } });
    }
    c.push_back({
        "PNG signature only", { 0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a },
         def, { status::bad_header }
    });
    c.push_back({
        "JPEG SOI only", { 0xff, 0xd8, 0xff },
         def, { status::bad_header }
    });
    c.push_back({
        "PNG width 0", png_header(0, 64, 16), def, { status::bad_header, status::empty }
    });
    c.push_back({ "PNG 10000x10000 (100 MP) header", png_header(10000, 10000, 16), def, { status::too_many_pixels } });
    c.push_back({
        "PNG 50000x10 header", png_header(50000, 10, 16), def, { status::bad_header, status::side_too_large }
    });
    c.push_back({ "PNG 2000x2000, garbage IDAT", png_header(2000, 2000, 64), def, { status::decode_failed } });
    const std::vector<uint8_t> jpg = read_file(images_dir + "/test-640x488.jpg");
    const std::vector<uint8_t> png = read_file(images_dir + "/photo-224.png");
    if (jpg.size() > 1000 && png.size() > 1000) {
        c.push_back({
            "JPEG truncated to 100 bytes",
            std::vector<uint8_t>(jpg.begin(), jpg.begin() + 100),
            def,
            { status::bad_header, status::decode_failed }
        });
        c.push_back({
            "JPEG truncated to half",
            std::vector<uint8_t>(jpg.begin(), jpg.begin() + jpg.size() / 2),
            def,
            { status::decode_failed, status::ok }
        });
        c.push_back({ "PNG truncated to half",
                      std::vector<uint8_t>(png.begin(), png.begin() + png.size() / 2),
                      def,
                      { status::decode_failed } });
        c.push_back(
            { "JPEG frame 9000x9000 (81 MP)", jpeg_resized_header(jpg, 9000, 9000), def, { status::too_many_pixels } });
        c.push_back({
            "JPEG frame 0x488",
            jpeg_resized_header(jpg, 0, 488),
            def,
            { status::bad_header, status::empty, status::decode_failed }
        });
        limits small;
        small.max_bytes = 1000;
        c.push_back({ "a PNG over a 1000-byte limit", png, small, { status::too_many_bytes } });
        limits narrow;
        narrow.max_side = 100;
        c.push_back({ "a 224-wide PNG over a 100 side limit", png, narrow, { status::side_too_large } });
        c.push_back({ "the 640x488 JPEG", jpg, def, { status::ok } });
        c.push_back({ "the 224x224 PNG", png, def, { status::ok } });
    }
    int fails = 0;
    for (const cas & k : c) {
        prepared     p;
        const status s  = prepare(k.bytes.data(), k.bytes.size(), k.lim, p);
        const bool   ok = std::find(k.want.begin(), k.want.end(), s) != k.want.end();
        std::printf("limit %-36s -> %-45s %s\n", k.name.c_str(), status_text(s), ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    const bool cnt = check_count(8, def) == status::ok && check_count(9, def) == status::too_many_images;
    std::printf("limit %-36s -> 8 ok, 9 %-38s %s\n", "image count", status_text(check_count(9, def)),
                cnt ? "PASS" : "FAIL");
    fails += !cnt;

    // the size rule's branches (the reference's numbers, VIS1's semantics)
    const struct {
        int32_t w, h, tw, th;
    } sz[] = {
        { 1920, 1080,  1920, 1088  },
        { 4032, 3024,  2336, 1760  },
        { 768,  768,   768,  768   },
        { 640,  488,   640,  480   },
        { 10,   10,    96,   96    },
        { 1,    10000, 32,   10016 },
        { 0,    5,     0,    0     }
    };

    for (const auto & z : sz) {
        const size2 t  = target_size(z.w, z.h);
        const bool  ok = t.w == z.tw && t.h == z.th;
        std::printf("size  %5dx%-5d -> %5dx%-5d (want %dx%d) %s\n", z.w, z.h, t.w, t.h, z.tw, z.th,
                    ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    std::printf("limits: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

int run_fuzz(const std::vector<std::string> & files, int32_t iters, uint32_t seed) {
    std::mt19937                   g(seed);
    std::map<std::string, int64_t> count;
    limits                         lim;
    lim.max_decoded = 4000000;  // each mutated decode stays small
    for (const std::string & f : files) {
        const std::vector<uint8_t> src = read_file(f);
        if (src.size() < 64) {
            std::printf("fuzz: cannot read %s\n", f.c_str());
            return 1;
        }
        for (int32_t i = 0; i < iters; ++i) {
            std::vector<uint8_t> m = src;
            switch (g() % 5) {
                case 0:  // bit flips
                    for (uint32_t k = 0, n = 1 + g() % 16; k < n; ++k) {
                        m[g() % m.size()] ^= uint8_t(1u << (g() % 8));
                    }
                    break;
                case 1:  // byte overwrites in the first 1 KB (the headers)
                    for (uint32_t k = 0, n = 1 + g() % 8; k < n; ++k) {
                        m[g() % std::min<size_t>(m.size(), 1024)] = uint8_t(g());
                    }
                    break;
                case 2:  // truncation
                    m.resize(1 + g() % m.size());
                    break;
                case 3:
                    {  // a random block repeated
                        const size_t a = g() % m.size(), n = std::min<size_t>(1 + g() % 4096, m.size() - a);
                        m.insert(m.begin() + int64_t(g() % m.size()), src.begin() + int64_t(a),
                                 src.begin() + int64_t(a + n));
                        break;
                    }
                default:  // a random run zeroed
                    for (size_t a = g() % m.size(), n = 1 + g() % 256, k = 0; k < n && a + k < m.size(); ++k) {
                        m[a + k] = 0;
                    }
                    break;
            }
            prepared     p;
            const status s = prepare(m.data(), m.size(), lim, p);
            if (s == status::ok &&
                (p.size.w % align || p.size.h % align || p.rows.size() != size_t(p.gw) * p.gh * patch_vals)) {
                std::printf("fuzz: an accepted image has a bad shape\n");
                return 1;
            }
            count[status_text(s)]++;
        }
    }
    for (const auto & k : count) {
        std::printf("fuzz %-45s %lld\n", k.first.c_str(), (long long) k.second);
    }
    std::printf("fuzz: %d files x %d mutations, no fault: PASS\n", int(files.size()), iters);
    return 0;
}

int run_range(const std::string & mmproj) {
    host_weights w;
    std::string  err;
    if (!load_host(mmproj, w, err)) {
        std::printf("range: %s\n", err.c_str());
        return 1;
    }
    range_report r;
    const bool   ok = range_check(w, r);
    std::printf(
        "range: profile '%s'; bounds for any image: LayerNorm out %.1f (block %d), q/k/v %.1f (block %d), "
        "FFN GELU %.1f (block %d), merger GELU %.1f; FP16 max 65504: %s\n",
        vision_profiles[w.profile].name, r.ln, r.ln_blk, r.qkv, r.qkv_blk, r.gelu, r.gelu_blk, r.merger_gelu,
        ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int run_ref64(const std::string &              mmproj,
              const std::vector<std::string> & images,
              const std::string &              ref_dir,
              int32_t                          threads) {
    host_weights w;
    std::string  err;
    if (!load_host(mmproj, w, err)) {
        std::printf("ref64: %s\n", err.c_str());
        return 1;
    }
    for (const std::string & path : images) {
        const std::vector<uint8_t> bytes = read_file(path);
        prepared                   p;
        if (prepare(bytes.data(), bytes.size(), limits(), p) != status::ok) {
            std::printf("ref64 %s: prepare failed\n", path.c_str());
            return 1;
        }
        const std::vector<double> out = ref::encoder(w, p, threads);
        const int32_t             T   = p.nx * p.ny;
        // keep the FP64 rows for the GPU test (F64 as stored)
        {
            std::ofstream f(ref_dir + "/" + base_of(path) + ".fp64.bin", std::ios::binary);
            f.write(reinterpret_cast<const char *>(out.data()), int64_t(out.size() * sizeof(double)));
        }
        for (const char * arm : { "rows", "fa.rows" }) {
            dump d;
            if (!read_dump(ref_dir + "/" + base_of(path) + "." + arm + ".bin", d) || d.dims[0] != out_width ||
                d.dims[1] != T) {
                std::printf("ref64 %s: no reference %s\n", base_of(path).c_str(), arm);
                continue;
            }
            const row_err e = compare_rows(d.f32.data(), out.data(), T, out_width);
            std::printf(
                "ref64 %-16s reference (%s attention) against FP64: per-row rel L2 max %.3e (row %d) mean %.3e, "
                "max abs / max %.3e\n",
                base_of(path).c_str(), std::strcmp(arm, "rows") == 0 ? "plain" : "flash", e.max_rel, e.worst_row,
                e.mean_rel, e.max_abs_over_max);
        }
        dump a, b;
        if (read_dump(ref_dir + "/" + base_of(path) + ".rows.bin", a) &&
            read_dump(ref_dir + "/" + base_of(path) + ".fa.rows.bin", b) && a.f32.size() == b.f32.size()) {
            const row_err e = compare_rows(b.f32.data(), a.f32.data(), T, out_width);
            std::printf("ref64 %-16s reference's own spread (flash against plain): max %.3e mean %.3e\n",
                        base_of(path).c_str(), e.max_rel, e.mean_rel);
        }
        std::printf("ref64 %-16s %d tokens\n", base_of(path).c_str(), T);
    }
    return 0;
}

namespace {

std::vector<double> per_row(const std::vector<double> & a, const std::vector<double> & ref, int32_t rows, int32_t n) {
    std::vector<double> e(static_cast<size_t>(rows));
    for (int32_t r = 0; r < rows; ++r) {
        double num = 0, den = 0;
        for (int32_t c = 0; c < n; ++c) {
            const double x = a[size_t(r) * n + c], y = ref[size_t(r) * n + c];
            num += (x - y) * (x - y);
            den += y * y;
        }
        e[size_t(r)] = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    }
    return e;
}

void print_rows(const char * img, const char * what, const std::vector<double> & e) {
    std::vector<int32_t> o(e.size());
    for (size_t i = 0; i < o.size(); ++i) {
        o[i] = int32_t(i);
    }
    std::sort(o.begin(), o.end(), [&](int32_t x, int32_t y) { return e[size_t(x)] > e[size_t(y)]; });
    double mean = 0;
    for (double v : e) {
        mean += v;
    }
    mean /= double(std::max<size_t>(1, e.size()));
    std::printf("ref16 %-16s %-34s per-row rel L2 max %.3e mean %.3e; worst rows", img, what,
                e.empty() ? 0.0 : e[size_t(o[0])], mean);
    for (size_t i = 0; i < std::min<size_t>(6, o.size()); ++i) {
        std::printf(" %d:%.3f", o[i], e[size_t(o[i])]);
    }
    std::printf("\n");
}

std::vector<double> read_f64(const std::string & path, size_t n) {
    std::vector<double> v(n);
    std::ifstream       f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char *>(v.data()), int64_t(n * sizeof(double)))) {
        v.clear();
    }
    return v;
}

}  // namespace

int run_ref16(const std::string &              mmproj,
              const std::vector<std::string> & images,
              const std::string &              ref_dir,
              int32_t                          threads) {
    host_weights w;
    std::string  err;
    if (!load_host(mmproj, w, err)) {
        std::printf("ref16: %s\n", err.c_str());
        return 1;
    }
    for (const std::string & path : images) {
        const std::vector<uint8_t> bytes = read_file(path);
        prepared                   p;
        if (prepare(bytes.data(), bytes.size(), limits(), p) != status::ok) {
            std::printf("ref16 %s: prepare failed\n", path.c_str());
            return 1;
        }
        const int32_t     T = p.nx * p.ny;
        const size_t      n = size_t(T) * out_width;
        const std::string b = ref_dir + "/" + base_of(path);
        ref::set_fp16_storage(true);
        const std::vector<double> emu = ref::encoder(w, p, threads);
        ref::set_fp16_storage(false);
        {
            std::ofstream f(b + ".fp16e.bin", std::ios::binary);
            f.write(reinterpret_cast<const char *>(emu.data()), int64_t(emu.size() * sizeof(double)));
        }
        const std::string         name = base_of(path);
        const char *              img  = name.c_str();
        const std::vector<double> r64  = read_f64(b + ".fp64.bin", n);
        if (!r64.empty()) {
            print_rows(img, "FP16 storage against FP64", per_row(emu, r64, T, out_width));
        }
        for (const char * arm : { "rows", "gpu" }) {
            std::vector<double> o;
            if (std::strcmp(arm, "gpu") == 0) {
                std::vector<float> g(n);
                std::ifstream      f(b + ".gpu.bin", std::ios::binary);
                if (f.read(reinterpret_cast<char *>(g.data()), int64_t(n * 4))) {
                    o.assign(g.begin(), g.end());
                }
            } else {
                dump d;
                if (read_dump(b + ".rows.bin", d) && d.dims[0] == out_width && d.dims[1] == T) {
                    o.assign(d.f32.begin(), d.f32.end());
                }
            }
            if (o.empty()) {
                continue;
            }
            const bool gpu = std::strcmp(arm, "gpu") == 0;
            if (!r64.empty()) {
                print_rows(img, gpu ? "GPU against FP64" : "reference engine against FP64",
                           per_row(o, r64, T, out_width));
            }
            print_rows(img, gpu ? "GPU against FP16 storage" : "reference engine against FP16 storage",
                       per_row(o, emu, T, out_width));
        }
        std::printf("ref16 %-16s %d tokens\n", img, T);
    }
    return 0;
}

}  // namespace fn::vision::test

namespace fn::vision::test {

int run_semantics(const std::string & mmproj, const std::string & image, const std::string & prefix) {
    host_weights               w;
    std::string                err;
    prepared                   p;
    const std::vector<uint8_t> bytes = read_file(image);
    if (!load_host(mmproj, w, err) || prepare(bytes.data(), bytes.size(), limits(), p) != status::ok) {
        std::printf("semantics: %s\n", err.empty() ? "prepare failed" : err.c_str());
        return 1;
    }
    const int32_t P = p.gw * p.gh, T = P / 4;
    const double  eps = vision_profiles[w.profile].eps;
    const auto    get = [&](const char * name, dump & d) {
        if (!read_dump(prefix + "." + name + ".bin", d)) {
            std::printf("semantics: no dump %s\n", name);
            return false;
        }
        return true;
    };
    const auto as64 = [](const dump & d) {
        return std::vector<double>(d.f32.begin(), d.f32.end());
    };
    const auto line = [](const char * step, const row_err & e) {
        std::printf("semantics %-44s per-row rel L2 max %.3e mean %.3e\n", step, e.max_rel, e.mean_rel);
    };
    dump pe, l1, qc, kc, vc, qr, kr, kqv, fi, fn_, ub, ge, lo26, rows;
    if (!get("inp_pos_emb", pe) || !get("ln1-0", l1) || !get("Qcur-0", qc) || !get("Kcur-0", kc) ||
        !get("Vcur-0", vc) || !get("Qcur_rope-0", qr) || !get("Kcur_rope-0", kr) || !get("kqv_out-0", kqv) ||
        !get("ffn_inp-0", fi) || !get("ffn_inp_normed-0", fn_) || !get("ffn_up_b-0", ub) || !get("ffn_gelu-0", ge) ||
        !get("layer_out-26", lo26) || !get("rows", rows)) {
        return 1;
    }
    // the patch embedding, the merge order and the position table, from the same pixels
    const auto x0 = ref::patch_embed(p.rows.data(), P, w.patch, w.patch_b, ref::pos_resize(w.pos, p.gw, p.gh));
    line("patch embedding + bias + position table", compare_rows(pe.f32.data(), x0.data(), P, width));
    const block_host & b = w.blk[0];
    const auto         l = ref::layernorm(as64(pe), P, width, b.ln1_w, b.ln1_b, eps);
    line("ln1 (block 0) of the reference's input", compare_rows(l1.f32.data(), l.data(), P, width));
    const auto qkv = ref::product(as64(l1), P, width, b.qkv, qkv_width, b.qkv_b, 8);
    {
        std::vector<double> q(size_t(P) * width), k(q.size()), v(q.size());
        for (int32_t r = 0; r < P; ++r) {
            for (int32_t c = 0; c < width; ++c) {
                q[size_t(r) * width + c] = qkv[size_t(r) * qkv_width + c];
                k[size_t(r) * width + c] = qkv[size_t(r) * qkv_width + width + c];
                v[size_t(r) * width + c] = qkv[size_t(r) * qkv_width + 2 * width + c];
            }
        }
        line("qkv product + bias (q) (BF16 activations there)", compare_rows(qc.f32.data(), q.data(), P, width));
        line("qkv product + bias (k)", compare_rows(kc.f32.data(), k.data(), P, width));
        line("qkv product + bias (v)", compare_rows(vc.f32.data(), v.data(), P, width));
    }
    {
        std::vector<double> q = as64(qc), k = as64(kc);
        for (int32_t r = 0; r < P; ++r) {
            const int32_t g = r / 4, s = r % 4;
            const int32_t gx = 2 * (g % (p.gw / 2)) + s % 2, gy = 2 * (g / (p.gw / 2)) + s / 2;
            for (int32_t h = 0; h < heads; ++h) {
                ref::rope(&q[(size_t(r) * heads + h) * head_dim], gy, gx);
                ref::rope(&k[(size_t(r) * heads + h) * head_dim], gy, gx);
            }
        }
        line("vision rotary on the reference's q", compare_rows(qr.f32.data(), q.data(), P, width));
        line("vision rotary on the reference's k", compare_rows(kr.f32.data(), k.data(), P, width));
    }
    {
        std::vector<double> q(size_t(P) * head_dim), k(q.size()), v(q.size()), o(q.size()), att(size_t(P) * width);
        for (int32_t h = 0; h < heads; ++h) {
            for (int32_t r = 0; r < P; ++r) {
                for (int32_t c = 0; c < head_dim; ++c) {
                    const size_t i              = (size_t(r) * heads + h) * head_dim + c;
                    q[size_t(r) * head_dim + c] = qr.f32[i];
                    k[size_t(r) * head_dim + c] = kr.f32[i];
                    v[size_t(r) * head_dim + c] = vc.f32[i];
                }
            }
            ref::attention(q.data(), k.data(), v.data(), P, o.data());
            for (int32_t r = 0; r < P; ++r) {
                for (int32_t c = 0; c < head_dim; ++c) {
                    att[size_t(r) * width + h * head_dim + c] = o[size_t(r) * head_dim + c];
                }
            }
        }
        line("attention on the reference's q, k and v", compare_rows(kqv.f32.data(), att.data(), P, width));
    }
    const auto l2 = ref::layernorm(as64(fi), P, width, b.ln2_w, b.ln2_b, eps);
    line("ln2 (block 0) of the reference's input", compare_rows(fn_.f32.data(), l2.data(), P, width));
    {
        std::vector<double> g = as64(ub);
        for (double & e : g) {
            e = ref::gelu(e);
        }
        line("GELU (tanh) of the reference's up + bias", compare_rows(ge.f32.data(), g.data(), P, ffn));
    }
    const auto m = ref::merger(w, as64(lo26), P, eps, 8);
    line("post_ln, merge and merger of block 26's output", compare_rows(rows.f32.data(), m.data(), T, out_width));
    return 0;
}

}  // namespace fn::vision::test
