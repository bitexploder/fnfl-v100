#include "fn/runtime/image_rows.h"

#include <algorithm>

namespace fn::runtime {

namespace {

constexpr int32_t qsa_block = 4;  // fn::qsa::ratio (a host-only file: the constant, not the CUDA header)

uint64_t fnv_u32(uint64_t h, uint32_t v) {
    for (int32_t b = 0; b < 4; ++b) {
        h = (h ^ ((v >> (8 * b)) & 0xffu)) * 0x100000001b3ull;
    }
    return h;
}

// the shift an image adds to the offset of every later cell
int32_t shift_of(const image_span & s) {
    return std::max(s.nx, s.ny) - s.nx * s.ny;
}

}  // namespace

bool image_rows::set(const int32_t * tok, int64_t n, const std::vector<image_span> & spans, std::string & err) {
    int64_t prev = 0;
    for (size_t i = 0; i < spans.size(); ++i) {
        const image_span & s = spans[i];
        if (s.nx < 1 || s.ny < 1 || s.tokens() > image_max_tokens || s.cell < prev || s.end() > n) {
            err = "image " + std::to_string(i) + ": cells " + std::to_string(s.cell) + " .. " +
                  std::to_string(s.end()) + " (" + std::to_string(s.nx) + " x " + std::to_string(s.ny) +
                  ") outside the prompt's " + std::to_string(n) + " cells or overlapping the image before";
            return false;
        }
        for (int64_t c = s.cell; c < s.end(); ++c) {
            if (tok == nullptr || tok[c] != image_token) {
                err = "image " + std::to_string(i) + ": cell " + std::to_string(c) + " holds token " +
                      std::to_string(tok == nullptr ? -1 : tok[c]) + ", not the image token " +
                      std::to_string(image_token);
                return false;
            }
        }
        prev = s.end();
    }
    s_ = spans;
    return true;
}

bool image_rows::cuts(int64_t cell) const {
    for (const image_span & s : s_) {
        if (cell > s.cell && cell < s.end()) {
            return true;
        }
    }
    return false;
}

void image_rows::keep(int64_t n) {
    while (!s_.empty() && s_.back().end() > n) {
        s_.pop_back();
    }
}

int32_t image_rows::offset_at(int64_t cell) const {
    int32_t off = 0;
    for (const image_span & s : s_) {
        if (s.end() > cell) {
            break;
        }
        off += shift_of(s);
    }
    return off;
}

bool image_rows::any_image(int64_t lo, int64_t hi) const {
    for (const image_span & s : s_) {
        if (s.cell < hi && s.end() > lo) {
            return true;
        }
    }
    return false;
}

void image_rows::planes(int64_t lo, int32_t rows, int32_t * out) const {
    // walk the cells once with the images in order: the offset changes only at an image's end
    size_t  i   = 0;
    int32_t off = 0;
    while (i < s_.size() && s_[i].end() <= lo) {
        off += shift_of(s_[i]);
        ++i;
    }
    for (int32_t r = 0; r < rows; ++r) {
        const int64_t c = lo + r;
        while (i < s_.size() && s_[i].end() <= c) {
            off += shift_of(s_[i]);
            ++i;
        }
        int32_t * o = out + int64_t(r) * 4;
        if (i < s_.size() && c >= s_[i].cell) {
            const image_span & s  = s_[i];
            const int64_t      k  = c - s.cell;
            const int32_t      p0 = static_cast<int32_t>(s.cell) + off;
            o[0]                  = p0;
            o[1]                  = p0 + static_cast<int32_t>(k / s.nx);
            o[2]                  = p0 + static_cast<int32_t>(k % s.nx);
            o[3]                  = 0;
        } else {
            const int32_t p = static_cast<int32_t>(c) + off;
            o[0] = o[1] = o[2] = o[3] = p;
        }
    }
}

void image_rows::chunk(int64_t start, int32_t rows, image_chunk & out) const {
    out.planes.clear();
    out.bplanes.clear();
    out.runs.clear();
    // a block the chunk completes may begin before it (start not a block start): its first row counts
    const int64_t b_lo = start / qsa_block, b_hi = (start + rows) / qsa_block;  // blocks b_lo .. b_hi - 1
    out.images   = any_image(b_lo * qsa_block, start + rows);
    out.rope_off = out.images ? 0 : offset_at(start);
    if (!out.images) {
        return;
    }
    out.planes.resize(size_t(rows) * 4);
    planes(start, rows, out.planes.data());
    out.bplanes.resize(size_t(std::max<int64_t>(b_hi - b_lo, 0)) * 4);
    for (int64_t b = b_lo; b < b_hi; ++b) {
        planes(b * qsa_block, 1, out.bplanes.data() + (b - b_lo) * 4);
    }
    for (size_t i = 0; i < s_.size(); ++i) {
        const image_span & s  = s_[i];
        const int64_t      lo = std::max(s.cell, start), hi = std::min(s.end(), start + rows);
        if (lo < hi) {
            out.runs.push_back({ static_cast<int32_t>(lo - start), static_cast<int32_t>(hi - lo),
                                 static_cast<int32_t>(i), lo - s.cell });
        }
    }
}

uint64_t image_rows::key(const int32_t * tok, int64_t n) const {
    uint64_t h = 0xcbf29ce484222325ull;
    size_t   i = 0;
    for (int64_t c = 0; c < n; ++c) {
        if (i < s_.size() && s_[i].cell == c) {
            const image_span & s = s_[i];
            h = fnv_u32(fnv_u32(h, static_cast<uint32_t>(s.digest)), static_cast<uint32_t>(s.digest >> 32));
            h = fnv_u32(fnv_u32(h, static_cast<uint32_t>(s.nx)), static_cast<uint32_t>(s.ny));
            ++i;
        }
        h = fnv_u32(h, static_cast<uint32_t>(tok[c]));
    }
    return h;
}

int64_t image_rows::common(const image_rows & o, int64_t n) const {
    int64_t m = n;
    size_t  i = 0;
    // the first image where the two differ (or one has an image the other lacks) ends the agreement at its start
    while (i < s_.size() || i < o.s_.size()) {
        const image_span * a = i < s_.size() ? &s_[i] : nullptr;
        const image_span * b = i < o.s_.size() ? &o.s_[i] : nullptr;
        if (a != nullptr && b != nullptr && *a == *b) {
            ++i;
            continue;
        }
        const int64_t first = std::min(a != nullptr ? a->cell : n, b != nullptr ? b->cell : n);
        m                   = std::min(m, first);
        break;
    }
    // and the prefix may not cut an image of either
    for (const image_rows * l : { this, &o }) {
        for (const image_span & s : l->s_) {
            if (m > s.cell && m < s.end()) {
                m = s.cell;
            }
        }
    }
    return std::max<int64_t>(m, 0);
}

bool image_rows::end_ok(int64_t n, std::string & err) const {
    // decode pools the block it completes at 4b + off(n): right when its first row b0 is text at the final offset
    const int64_t b0 = n / qsa_block * qsa_block;
    if (b0 == n) {
        return true;
    }
    for (const image_span & s : s_) {
        if (s.end() > b0 && s.cell < n) {
            // the shortest valid end is the image's end rounded up to a whole block
            const int64_t need = (s.end() + qsa_block - 1) / qsa_block * qsa_block - s.end();
            err = n <= s.end() ? "the prompt ends inside the image at cell " + std::to_string(s.cell) :
                                 "the prompt ends " + std::to_string(n - s.end()) + " tokens after the image at cell " +
                                     std::to_string(s.cell) + ", inside the image's last QSA block; at least " +
                                     std::to_string(need) + " tokens must follow it";
            return false;
        }
    }
    return true;
}

}  // namespace fn::runtime
