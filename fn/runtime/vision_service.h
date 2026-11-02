#pragma once

// VIS1 Stage B: the vision service, the prompt path's image feed (prefill.h pf_images). It holds the projector on both
// cards of the box that embeds (card 0: blocks 0 .. 13 and the patch embedding; card 1: blocks 14 .. 26 and the
// merger) and encodes an image when a chunk first holds its rows, into its read lane's encoder rows on card 1. Each
// chunk's image runs are then copied out of them, card c's half of the 2,560 columns into card c's supplied rows
// (packed in run order), and the chunk's rope planes and block planes uploaded, so the chunk's embed_supply and rotary
// sites read them. All of its memory is the model's vision classes (fn/model/state.h vision_classes, reserved by the
// loader for load_params::mmproj; on the split's front the work class is the unused other-reader arena).
//
// Slices (the ticket's writers rule). fill queues each image its chunk reaches that the lane does not hold; one
// encoder serves both lanes, so the queue's images encode one after another. On the multi-slot engine's job steps
// (sliced) the chunk's first steps are encode_step calls, each a slice of encoder blocks completed before it returns,
// so other slots' rounds run between them; the chunk's first layer group follows its last image. Otherwise fill
// encodes before returning. A failed or abandoned read drops its lane's queue (forget).
//
// Ordering (the host waits for a plane upload slot's previous copy, embed's patch staging and each slice's end):
//   card 0's encoder stream: [wait: card 1 copied the previous residual] embed, blocks 0 .. 13, record `blocks`;
//   card 1's encoder stream: wait `blocks`, copy the residual, record `copied`, blocks 14 .. 26, the merger into the
//     lane's encoder rows; then [wait: the target streams' tails, recorded at fill, so the lane's previous chunk has
//     read its supplied rows] the run's copies into both cards' supplied rows, and after the chunk's last image,
//     record `supplied`;
//   each target stream: wait `supplied`, then the plane uploads (stream order after the previous chunk's readers).
// A lane's encoder rows hold the last image it encoded (its slot, cell and digest); a chunk whose image is held
// copies at fill without encoding again, so an image whose rows straddle two chunks is encoded once unless another
// image came between them in that lane.

#include "fn/model/model.h"
#include "fn/model/prompt.h"
#include "fn/runtime/image_rows.h"
#include "fn/runtime/prefill.h"
#include "fn/runtime/target.h"
#include "fn/vision/encoder.h"
#include "fn/vision/weights.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace fn::runtime {

class vision_service final : public pf_images {
  public:
    vision_service() = default;
    ~vision_service() override;
    vision_service(const vision_service &)             = delete;
    vision_service & operator=(const vision_service &) = delete;

    // the projector file at path into m's vision classes: read and resolved (weights.h load_host), the FP16 range
    // proof, each card's share converted and placed, the two encoders on their class memory. False (err) when the
    // classes are missing (the load reserved none) or differ from the plan, or the file is refused
    bool init(model::model & m, const std::string & path, std::string & err);
    // the split's back: the planes and block planes only (it embeds nothing; model::vision_params::planes_only)
    bool init_planes(model::model & m, std::string & err);
    // tests (split-test's image case): an image's rows [nx ny][2,560] F32 from rows(span, out) in place of the encoder
    // (the vision classes reserved without weights or scratch)
    using synthetic_rows = std::function<void(const image_span &, float *)>;
    bool init_synthetic(model::model & m, const synthetic_rows & rows, std::string & err);
    // after init or init_synthetic, one box: the image token's embedding row on each card (the weights class's token
    // row, state.h vtr_*), embedded with t's token_embd slices as embed_rows and embed_join embed a prompt row, so the
    // draft's input at the supplied rows is the token's, bitwise (prompt_chunk::draft_e). Synchronous. Nothing when
    // the load reserved no row; false (err) then if `drafts` (the box drafts over the rows it supplies)
    bool token_row(target & t, bool drafts, std::string & err);

    bool fill(target & t, int32_t lane, const image_chunk & h, model::prompt_chunk & c, bool sliced, std::string & err)
        override;
    bool pending(int32_t lane) const override;
    bool encode_step(int32_t lane, int32_t blocks, std::string & err) override;
    void forget(int32_t lane) override;

    // fill's work for slot `slot` with ledger `rows`, its prepared images px (px(i): image i's, or null) and its
    // compute streams (card 0, card 1); fill passes the target's (and vis-test its own)
    using pixels = std::function<std::shared_ptr<const vision::prepared>(size_t)>;
    bool fill_rows(int32_t               slot,
                   const image_rows &    rows,
                   const pixels &        px,
                   void * const          streams[2],
                   int32_t               lane,
                   const image_chunk &   h,
                   model::prompt_chunk & c,
                   bool                  sliced,
                   std::string &         err);

    int32_t lanes() const { return lanes_; }

    int64_t encodes() const { return encodes_; }  // images encoded since init

  private:
    struct held {
        int32_t  slot   = -1;
        int64_t  cell   = -1;
        uint64_t digest = 0;
        int32_t  nx = 0, ny = 0;
    };

    enum class mode : int32_t { encoder, planes, synthetic };

    // an image a lane's chunk waits for (fill, sliced or not), encoded by encode_step in queue order: its run of the
    // chunk and where that run's rows go in the lane's supply (packed), the chunk's target streams, and its slices
    struct queued {
        int32_t                                 lane = -1, slot = -1;
        image_span                              span;
        std::shared_ptr<const vision::prepared> img;
        image_run                               run;
        int64_t                                 done       = 0;
        void *                                  streams[2] = { nullptr, nullptr };
        int32_t                                 steps      = 0;
        double                                  wall_s     = 0;
    };

    bool queue_rows(int32_t               slot,
                    const image_rows &    rows,
                    const pixels &        px,
                    void * const          streams[2],
                    int32_t               lane,
                    const image_chunk &   h,
                    model::prompt_chunk & c,
                    std::string &         err);
    // the queue front's next encoder blocks, at most `budget` (next_ onward): the patch embedding before block 0, the
    // residual's copy to card 1 before block vision_split, the merger into the lane's encoder rows after the last
    bool advance(int32_t budget, std::string & err);
    bool synthetic(const image_span & s, int32_t lane, std::string & err);  // the tests' rows into the lane's
    bool copy_run(int32_t lane, const image_run & u, int64_t done, std::string & err);
    bool supplied(void * const streams[2], std::string & err);  // the target streams wait for the supply copies
    bool own_stream(std::string & err);
    bool attach(model::model & m, std::string & err);           // the lanes' parts and the events
    void release();

    model::model *         m_    = nullptr;
    mode                   mode_ = mode::encoder;
    synthetic_rows         synth_;
    void *                 s1_     = nullptr;  // card 1's copy stream (the encoder's; else own_s1_)
    void *                 own_s1_ = nullptr;
    vision::device_weights w_[2];
    vision::encoder        enc_[2];
    int32_t                lanes_        = 0;
    int32_t                lane_rows_[2] = { 0, 0 };
    held                   held_[2];
    float *                rows_[2]      = { nullptr, nullptr };  // lane: encoder rows (card 1)
    float *                supply_[2][2] = {
        { nullptr, nullptr },
        { nullptr, nullptr }
    };  // lane, card
    uint8_t * planes_[2][2] = {
        { nullptr, nullptr },
        { nullptr, nullptr }
    };  // lane, card (device)
    uint8_t * slots_[2][2] = {
        { nullptr, nullptr },
        { nullptr, nullptr }
    };  // lane, card (host, 2 each)
    int32_t                                      turn_[2]   = { 0, 0 };
    void *                                       ev_blocks_ = nullptr, *ev_copied_ = nullptr, *ev_supplied_ = nullptr;
    void *                                       ev_tail_[2]     = { nullptr, nullptr };
    void *                                       ev_up_[2][2][2] = {};  // lane, card, slot
    std::vector<model::prompt_chunk::supply_run> runs_[2];
    const float *      tok_full_[2] = { nullptr, nullptr };             // card: the image token's row (token_row)
    std::deque<queued> queue_;
    int32_t            next_    = 0;  // the queue front's next encoder block (0: not begun)
    int64_t            encodes_ = 0;
};

}  // namespace fn::runtime
