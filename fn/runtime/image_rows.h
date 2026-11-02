#pragma once

// VIS1 Stage B: a slot's image spans beside its tokens, and everything positional derived from them (host only).
//
// An image of nx x ny merged tokens occupies nx ny consecutive cells from its first cell c0, each holding the PLE image
// token (image_token, the file's qwen4exp.ple.image_token_id; the reference stores the same for embedding rows,
// ggml-vfast 66e11c3 src/llama-kv-cache.cpp:1262). Rotary planes (t, h, w, e), the reference's M-RoPE layout:
//   a text row at cell c:          (p, p, p, p), p = c + off(c);
//   image row k (grid x = k % nx, y = k / nx) of an image at c0, p0 = c0 + off(c0): (p0, p0 + y, p0 + x, 0);
//   off(c) = the sum over the images ending at or before c of max(nx, ny) - nx ny (text after an image resumes at
//   p0 + max(nx, ny)). With no image, every plane is the cell and off is 0: text keeps its bits.
// The QSA block summaries take their block's first row's planes; decode pools a block it completes at 4b + off(n), so a
// prompt whose last partial block holds an image row (its first row then an image row or text at an earlier offset)
// is refused (end_ok).

#include <cstdint>
#include <string>
#include <vector>

namespace fn::runtime {

constexpr int32_t image_token      = 248056;  // <|image_pad|>, the file's qwen4exp.ple.image_token_id
constexpr int32_t image_max_tokens = 4096;    // an image's merged tokens at most (the encoder's cap)

struct image_span {
    int64_t  cell   = 0;  // its first cell
    int32_t  nx     = 0;  // merged tokens across
    int32_t  ny     = 0;  // and down
    uint64_t digest = 0;  // its content (pixels and preprocessing), for keys

    int64_t tokens() const { return int64_t(nx) * ny; }

    int64_t end() const { return cell + tokens(); }

    bool operator==(const image_span & o) const {
        return cell == o.cell && nx == o.nx && ny == o.ny && digest == o.digest;
    }
};

// a chunk's run of image rows: chunk rows row0 .. row0 + n - 1 are tokens k0 .. k0 + n - 1 of image `image` (its index
// in the ledger; an image's rows may straddle two chunks)
struct image_run {
    int32_t row0  = 0;
    int32_t n     = 0;
    int32_t image = 0;
    int64_t k0    = 0;
};

// the positions of a prompt chunk of `rows` cells from `start` (image_rows::chunk). images: some QSA block the chunk
// completes or some row of it is an image row; then planes ([rows][4], every row) and bplanes (blocks start / 4 ..
// (start + rows) / 4 - 1, [blocks][4], each its first row's planes) are set, and runs lists the image rows. Otherwise
// every row and block reads cell + rope_off, and the arrays are empty
struct image_chunk {
    bool                   images   = false;
    int32_t                rope_off = 0;
    std::vector<int32_t>   planes;
    std::vector<int32_t>   bplanes;
    std::vector<image_run> runs;
};

class image_rows {
  public:
    // the prompt's spans for tokens tok[0, n): sorted, disjoint, inside [0, n), each 1 .. image_max_tokens tokens with
    // nx, ny >= 1, and every cell of each holding image_token. False (err set, nothing changed) otherwise
    bool set(const int32_t * tok, int64_t n, const std::vector<image_span> & spans, std::string & err);

    void clear() { s_.clear(); }

    bool empty() const { return s_.empty(); }

    const std::vector<image_span> & spans() const { return s_; }

    // cell lies strictly inside an image (a prefix of `cell` cells would cut it)
    bool     cuts(int64_t cell) const;
    // keep the images wholly in the first n cells (n must not cut one)
    void     keep(int64_t n);
    // off(c) above, for c not inside an image (c == an image's first cell is fine)
    int32_t  offset_at(int64_t cell) const;
    // some image row lies in [lo, hi)
    bool     any_image(int64_t lo, int64_t hi) const;
    // out[r][4] = the planes of cell lo + r, r < rows
    void     planes(int64_t lo, int32_t rows, int32_t * out) const;
    // FNV-1a over tok[0, n) as fn::runtime::token_key, with each image starting before n mixed in at its first cell
    // (digest, nx, ny) before that cell's token: equal token ids never mean equal prompts. No image: token_key's value
    uint64_t key(const int32_t * tok, int64_t n) const;
    // the longest prefix m <= n over which this ledger and o agree: no image of either starting before m differs,
    // and m cuts no image of either
    int64_t  common(const image_rows & o, int64_t n) const;
    // the chunk's positions (image_chunk above)
    void     chunk(int64_t start, int32_t rows, image_chunk & out) const;
    // the prompt of n cells may end here: its last partial QSA block (4 floor(n / 4) .. n - 1) holds no image row
    bool     end_ok(int64_t n, std::string & err) const;

  private:
    std::vector<image_span> s_;
};

}  // namespace fn::runtime
