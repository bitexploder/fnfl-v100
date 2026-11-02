#pragma once

// VIS1: the vision encoder on one card. Input: one prepared image (image.h); output: its embedding rows, F32
// [nx ny][2560], row t at merged grid (t / nx, t % nx), the (t, h, w) grid position (0, t / nx, t % nx). It knows
// nothing of the text model, the split or the slots; a caller can run the blocks of a range on each card (embed on
// the first, merge on the last) and copy the residual between them (residual(), P x 1152 F32).
//
// The scratch is one allocation, sized for max_patches and reused block by block, one image at a time:
//   x  F32 [P][1152]         the residual
//   h  FP16 [P][1152]        LayerNorm outputs; the attention output reuses it (h is dead after the qkv product)
//   qkv FP16 Q, K [16][Pp][72] and Vt [16][80][Pp]; the patch rows (F32 [P][768]) before block 0
//   f32 [chunk][4608]        a row chunk's product output (qkv, o, up, down, mm.0)
//   f16 [chunk][4608]        a row chunk's GELU output (the down and mm.2 inputs)
// The products are pd_tc over row chunks (FP16 operands, F32 accumulators, the weights' column scales).

#include "fn/vision/image.h"
#include "fn/vision/kernels.h"
#include "fn/vision/sizes.h"
#include "fn/vision/weights.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace fn::vision {

class encoder {
  public:
    encoder() = default;
    ~encoder();
    encoder(const encoder &)             = delete;
    encoder & operator=(const encoder &) = delete;

    // chunk: rows a product chunk, a multiple of 64 (tests use small chunks to cover the chunk loops). VIS1 Stage B:
    // with scratch (plan_scratch(max_patches, chunk).total device bytes on this card, 256-byte aligned) and staging
    // (patch_stage_bytes(max_patches) pinned host bytes; null on a card that never embeds) the encoder uses that memory
    // (a state class's) and allocates none; without, it allocates its own (the tests')
    bool init(int32_t       device,
              int32_t       max_patches,
              std::string & err,
              int32_t       chunk   = enc_chunk,
              void *        scratch = nullptr,
              float *       staging = nullptr);
    void release();

    // the patch rows to the card, then x = patch embedding + bias + resized position table (w.has_embed)
    bool embed(const device_weights & w, const prepared & img);
    // the split's entry on a later card: x = a residual (host or device pointer, F32 [gw gh][1152]) for a gw x gh grid
    bool begin_from_residual(const float * x, int32_t gw, int32_t gh);
    // blocks [first, last) of w's range over the current image
    bool run_blocks(const device_weights & w, int32_t first, int32_t last);
    // post_ln, the 2 x 2 merge (a reshape) and the merger MLP into out, F32 [P / 4][2560] on this card
    bool merge(const device_weights & w, float * out);
    // all three on this card
    bool encode(const device_weights & w, const prepared & img, float * out);
    bool sync();

    float * residual() const { return x_; }

    int32_t patches() const { return p_; }

    void * stream() const { return stream_; }

    size_t scratch_bytes() const { return plan_.total; }

    const char * error() const { return err_; }

  private:
    bool product(const uint16_t * a,
                 int32_t          rows,
                 int32_t          k,
                 const uint16_t * w,
                 const float *    cs,
                 int32_t          n,
                 float *          out,
                 int64_t          ld_out);
    bool fail(const char * what);
    bool shape(int32_t gw, int32_t gh);

    int32_t      device_ = -1;
    scratch_plan plan_;
    void *       arena_  = nullptr;
    float *      pinned_ = nullptr;  // the patch rows' host staging, max_patches x 768
    bool         owned_  = false;    // arena_ and pinned_ are the encoder's own (init without external memory)
    void *       stream_ = nullptr;
    float *      x_      = nullptr;
    uint16_t *   h_ = nullptr, *q_ = nullptr, *k_ = nullptr, *vt_ = nullptr, *f16_ = nullptr;
    float *      f32_ = nullptr;
    int32_t      p_ = 0, pp_ = 0, gw_ = 0, gh_ = 0;
    const char * err_ = nullptr;
};

}  // namespace fn::vision
