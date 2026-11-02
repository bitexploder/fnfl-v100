#include "fn/runtime/image_rows_test.h"

#include "fn/runtime/image_rows.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace fn::runtime {

namespace {

struct prompt {
    std::vector<int32_t>    tok;
    std::vector<image_span> spans;
};

// the reference's positions, token by token: [n][4]
std::vector<int32_t> walk(const prompt & p) {
    std::vector<int32_t> out(p.tok.size() * 4);
    int32_t              pos = 0;
    size_t               i   = 0;
    for (int64_t c = 0; c < int64_t(p.tok.size());) {
        if (i < p.spans.size() && p.spans[i].cell == c) {
            const image_span & s = p.spans[i];
            for (int32_t y = 0; y < s.ny; ++y) {
                for (int32_t x = 0; x < s.nx; ++x) {
                    int32_t * o = &out[size_t(c + int64_t(y) * s.nx + x) * 4];
                    o[0]        = pos;
                    o[1]        = pos + y;
                    o[2]        = pos + x;
                    o[3]        = 0;
                }
            }
            pos += std::max(s.nx, s.ny);
            c += s.tokens();
            ++i;
        } else {
            int32_t * o = &out[size_t(c) * 4];
            o[0] = o[1] = o[2] = o[3] = pos;
            ++pos;
            ++c;
        }
    }
    return out;
}

// FNV-1a over the tokens, as fn::runtime::token_key
uint64_t token_fnv(const std::vector<int32_t> & t, int64_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t v = static_cast<uint32_t>(t[size_t(i)]);
        for (int32_t b = 0; b < 4; ++b) {
            h = (h ^ ((v >> (8 * b)) & 0xffu)) * 0x100000001b3ull;
        }
    }
    return h;
}

prompt random_prompt(std::mt19937 & g, int32_t n_images) {
    prompt                                 p;
    std::uniform_int_distribution<int32_t> text(0, 40), side(1, 24), tid(0, 248000);
    for (int32_t k = 0; k < n_images; ++k) {
        for (int32_t t = text(g); t > 0; --t) {
            p.tok.push_back(tid(g));
        }
        image_span s;
        s.cell   = int64_t(p.tok.size());
        s.nx     = side(g);
        s.ny     = side(g);
        s.digest = (uint64_t(g()) << 32) | g();
        p.spans.push_back(s);
        p.tok.insert(p.tok.end(), size_t(s.tokens()), image_token);
    }
    for (int32_t t = text(g); t > 0 || p.tok.empty(); --t) {  // at least one token
        p.tok.push_back(tid(g));
    }
    return p;
}

int fails = 0;

void check(bool ok, const std::string & what) {
    if (!ok) {
        std::printf("rows: FAIL %s\n", what.c_str());
        ++fails;
    }
}

}  // namespace

int image_rows_test(uint32_t seed) {
    fails = 0;
    std::mt19937 g(seed);
    std::string  err;

    // planes and offsets against the walk, every window of random prompts
    int32_t windows = 0;
    for (int32_t trial = 0; trial < 200; ++trial) {
        const prompt               p   = random_prompt(g, trial % 5);
        const std::vector<int32_t> ref = walk(p);
        image_rows                 l;
        if (!l.set(p.tok.data(), int64_t(p.tok.size()), p.spans, err)) {
            check(false, "set refused a valid prompt: " + err);
            continue;
        }
        const int64_t                          n = int64_t(p.tok.size());
        std::uniform_int_distribution<int64_t> at(0, n - 1);
        for (int32_t w = 0; w < 8; ++w) {
            const int64_t        lo = at(g), hi = std::min<int64_t>(n, lo + 1 + at(g));
            std::vector<int32_t> pl(size_t(hi - lo) * 4);
            l.planes(lo, int32_t(hi - lo), pl.data());
            check(std::equal(pl.begin(), pl.end(), ref.begin() + lo * 4),
                  "planes of cells " + std::to_string(lo) + " .. " + std::to_string(hi));
            const bool overlap = std::any_of(p.spans.begin(), p.spans.end(),
                                             [&](const image_span & s) { return s.cell < hi && s.end() > lo; });
            check(l.any_image(lo, hi) == overlap,
                  "any_image of cells " + std::to_string(lo) + " .. " + std::to_string(hi));
            ++windows;
        }
        // chunks: rows and completed blocks at the walk's planes (cell + rope_off without images), runs on image cells
        for (int32_t w = 0; w < 8; ++w) {
            const int64_t lo   = at(g);
            const int32_t rows = int32_t(std::min<int64_t>(n, lo + 1 + at(g)) - lo);
            image_chunk   ch;
            l.chunk(lo, rows, ch);
            const int64_t b_lo = lo / 4, b_hi = (lo + rows) / 4;
            const bool    want = std::any_of(p.spans.begin(), p.spans.end(), [&](const image_span & s) {
                return s.cell < lo + rows && s.end() > b_lo * 4;
            });
            check(ch.images == want, "chunk images at " + std::to_string(lo));
            bool    ok  = true;
            int64_t imr = 0;
            for (int32_t r = 0; r < rows; ++r) {
                for (int32_t q = 0; q < 4; ++q) {
                    const int32_t got = ch.images ? ch.planes[size_t(r) * 4 + q] : int32_t(lo + r) + ch.rope_off;
                    ok                = ok && got == ref[size_t(lo + r) * 4 + q];
                }
                imr += p.tok[size_t(lo + r)] == image_token && l.any_image(lo + r, lo + r + 1) ? 1 : 0;
            }
            for (int64_t b = b_lo; b < b_hi; ++b) {
                for (int32_t q = 0; q < 4; ++q) {
                    const int32_t got = ch.images ? ch.bplanes[size_t(b - b_lo) * 4 + q] : int32_t(4 * b) + ch.rope_off;
                    ok                = ok && got == ref[size_t(4 * b) * 4 + q];
                }
            }
            int64_t covered = 0;
            for (const image_run & u : ch.runs) {
                const image_span & s = p.spans[size_t(u.image)];
                ok                   = ok && u.n > 0 && s.cell + u.k0 == lo + u.row0 && u.k0 + u.n <= s.tokens();
                covered += u.n;
            }
            ok = ok && covered == imr && (ch.images || ch.runs.empty());
            check(ok, "chunk of cells " + std::to_string(lo) + " .. " + std::to_string(lo + rows));
        }
        // the offset at every text cell and every image's first cell
        for (int64_t c = 0; c < n; ++c) {
            if (!l.cuts(c)) {
                check(l.offset_at(c) == ref[size_t(c) * 4] - int32_t(c), "offset at cell " + std::to_string(c));
            }
        }
        // no image: the key is token_key's; images: mixed in
        if (p.spans.empty()) {
            check(l.key(p.tok.data(), n) == token_fnv(p.tok, n), "key without images");
        } else {
            image_rows l2     = l;
            auto       spans2 = p.spans;
            spans2[0].digest ^= 1;
            check(l2.set(p.tok.data(), n, spans2, err), "set with another digest");
            check(l2.key(p.tok.data(), n) != l.key(p.tok.data(), n), "keys of different images differ");
            check(l2.key(p.tok.data(), p.spans[0].cell) == l.key(p.tok.data(), p.spans[0].cell),
                  "keys agree before the first image");
            check(l.common(l2, n) == p.spans[0].cell, "common prefix stops at the differing image");
            check(l.common(l, n) == n, "a ledger agrees with itself");
            // a cut inside an image moves the common prefix to its start
            const image_span & s = p.spans.back();
            if (s.tokens() > 1) {
                check(l.common(l, s.cell + 1) == s.cell, "common prefix never cuts an image");
                check(l.cuts(s.cell + 1) && !l.cuts(s.cell) && !l.cuts(s.end()), "cuts");
            }
            image_rows l3 = l;
            l3.keep(s.end() - 1);
            check(l3.spans().size() == p.spans.size() - 1, "keep drops the image it would cut");
        }
    }

    // set refuses bad ledgers
    {
        std::vector<int32_t> tok(40, 7);
        std::fill(tok.begin() + 10, tok.begin() + 22, image_token);
        image_rows l;
        check(l.set(tok.data(), 40,
                    {
                        { 10, 4, 3, 1 }
        },
                    err),
              "set a 4 x 3 image");
        check(!l.set(tok.data(), 40,
                     {
                         { 9, 4, 3, 1 }
        },
                     err),
              "set refuses a text cell inside the image");
        check(!l.set(tok.data(), 40,
                     {
                         { 10, 0, 3, 1 }
        },
                     err),
              "set refuses nx 0");
        check(!l.set(tok.data(), 40,
                     {
                         { 10, 2, 3, 1 },
                         { 14, 2, 3, 1 }
        },
                     err),
              "set refuses overlap");
        check(!l.set(tok.data(), 20,
                     {
                         { 10, 4, 3, 1 }
        },
                     err),
              "set refuses an image past the prompt");
        std::vector<int32_t> big(5000, image_token);
        check(!l.set(big.data(), 5000,
                     {
                         { 0, 65, 64, 1 }
        },
                     err),
              "set refuses 4,160 tokens");
        check(l.set(big.data(), 5000,
                    {
                        { 0, 64, 64, 1 }
        },
                    err),
              "set takes 4,096 tokens");
    }

    // the end rule, cell by cell
    {
        std::vector<int32_t> tok(64, 7);
        std::fill(tok.begin() + 13, tok.begin() + 21, image_token);  // 4 x 2 at 13 .. 20, ending at 21
        image_rows l;
        check(l.set(tok.data(), 64,
                    {
                        { 13, 4, 2, 1 }
        },
                    err),
              "set the end rule's image");
        for (int64_t n = 0; n <= 40; ++n) {
            // valid: no image row in 4 floor(n / 4) .. n - 1 (or n a whole block)
            const int64_t b0   = n / 4 * 4;
            const bool    want = b0 == n || !(b0 < 21 && n > 13);
            check(l.end_ok(n, err) == want, "end rule at n = " + std::to_string(n));
        }
        check(!l.end_ok(22, err) && l.end_ok(24, err) && l.end_ok(25, err), "end rule after the image");
    }

    std::printf(
        "rows: %d windows of random prompts against the token walk, offsets, keys, common prefix, chunks, end rule: "
        "%s\n",
        windows, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

}  // namespace fn::runtime
