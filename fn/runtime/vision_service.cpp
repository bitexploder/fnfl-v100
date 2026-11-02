#include "fn/runtime/vision_service.h"

#include "fn/kernels/embed/embed.h"
#include "fn/model/pass.h"
#include "fn/model/state.h"
#include "fn/runtime/prompt_plan.h"
#include "fn/vision/profile.h"
#include "fn/vision/sizes.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace fn::runtime {

namespace {

constexpr int32_t half_cols = vision::out_width / 2;  // 1,280: a card's columns of a supplied row

bool fail(std::string & err, const std::string & what) {
    err = "vision: " + what;
    return false;
}

bool ok(cudaError_t e, const char * what, std::string & err) {
    if (e == cudaSuccess) {
        return true;
    }
    return fail(err, std::string(what) + ": " + cudaGetErrorString(e));
}

cudaEvent_t ev(void * e) {
    return static_cast<cudaEvent_t>(e);
}

cudaStream_t st(void * s) {
    return static_cast<cudaStream_t>(s);
}

}  // namespace

vision_service::~vision_service() {
    release();
}

void vision_service::release() {
    if (m_ == nullptr) {
        return;
    }
    for (int32_t c = 0; c < 2; ++c) {
        enc_[c].release();  // synchronizes its stream
        w_[c].release();
    }
    if (own_s1_ != nullptr) {
        cudaSetDevice(m_->device[1]);
        cudaStreamSynchronize(st(own_s1_));
        cudaStreamDestroy(st(own_s1_));
    }
    own_s1_ = s1_ = nullptr;
    lanes_        = 0;
    queue_.clear();
    next_        = 0;
    tok_full_[0] = tok_full_[1] = nullptr;
    const auto destroy          = [&](void *& e, int32_t c) {
        if (e != nullptr) {
            cudaSetDevice(m_->device[c]);
            cudaEventDestroy(ev(e));
            e = nullptr;
        }
    };
    destroy(ev_blocks_, 0);
    destroy(ev_copied_, 1);
    destroy(ev_supplied_, 1);
    for (int32_t c = 0; c < 2; ++c) {
        destroy(ev_tail_[c], c);
        for (int32_t l = 0; l < 2; ++l) {
            for (int32_t s = 0; s < 2; ++s) {
                destroy(ev_up_[l][c][s], c);
            }
        }
    }
    m_ = nullptr;
}

bool vision_service::init(model::model & m, const std::string & path, std::string & err) {
    release();
    const model::reserved_class * wc = m.state_class("vision weights (VIS1)");
    const model::reserved_class * dc = m.state_class("vision work (VIS1)");
    const model::reserved_class * hc = m.state_class("vision staging (VIS1)");
    if (wc == nullptr || dc == nullptr || hc == nullptr) {
        return fail(err, "the vision classes are not reserved (the load had no projector: load_params::mmproj)");
    }
    using namespace fn::vision;
    const int64_t want_w[2] = { int64_t(weight_bytes(0, vision_split, true, false)),
                                int64_t(weight_bytes(vision_split, blocks, false, true)) };
    const int64_t want_s    = int64_t(plan_scratch(max_patches).total);
    for (int32_t c = 0; c < 2; ++c) {
        if (wc->plan.parts[0].bytes[c] != want_w[c] || dc->plan.parts[model::vp_scratch].bytes[c] != want_s ||
            int64_t(weight_stage_bytes(c == 0 ? 0 : vision_split, c == 0 ? vision_split : blocks, c == 1)) > want_s) {
            return fail(err, "the vision class differs from fn/vision/sizes.h's plan");
        }
    }
    if (hc->plan.parts[model::vs_patch].bytes[0] != int64_t(patch_stage_bytes(max_patches))) {
        return fail(err, "the vision staging class differs from fn/vision/sizes.h's plan");
    }
    host_weights hw;
    range_report rr;
    if (!load_host(path, hw, err)) {
        return fail(err, path + ": " + err);
    }
    if (!range_check(hw, rr) || !rr.ok) {
        return fail(err, path + ": the FP16 range proof failed (an operand bound reaches 65,504)");
    }
    m_ = &m;
    for (int32_t c = 0; c < 2; ++c) {
        const int32_t b0 = c == 0 ? 0 : vision_split, b1 = c == 0 ? vision_split : blocks;
        void *        scratch = dc->part_ptr(model::vp_scratch, c);
        if (!w_[c].upload(hw, m.device[c], b0, b1, c == 0, c == 1, err, wc->part_ptr(0, c), scratch) ||
            !enc_[c].init(m.device[c], max_patches, err, enc_chunk, scratch,
                          c == 0 ? reinterpret_cast<float *>(hc->part_ptr(model::vs_patch, 0)) : nullptr)) {
            const std::string e = err;
            release();
            return fail(err, "card " + std::to_string(c) + ": " + e);
        }
    }
    hw    = host_weights();  // the host copy is no longer needed
    mode_ = mode::encoder;
    s1_   = enc_[1].stream();
    if (!attach(m, err)) {
        return false;
    }
    std::fprintf(stderr,
                 "FNFL vision: %s: weights %.1f + %.1f MiB, scratch %.1f MiB a card, %d read lane%s (%d and %d rows)\n",
                 path.c_str(), double(want_w[0]) / 1048576.0, double(want_w[1]) / 1048576.0, double(want_s) / 1048576.0,
                 lanes_, lanes_ > 1 ? "s" : "", lane_rows_[0], lane_rows_[1]);
    return true;
}

bool vision_service::init_planes(model::model & m, std::string & err) {
    release();
    if (m.state_class("vision work (VIS1)") == nullptr || m.state_class("vision staging (VIS1)") == nullptr) {
        return fail(err, "the vision classes are not reserved (the split's back takes planes only)");
    }
    m_    = &m;
    mode_ = mode::planes;
    return own_stream(err) && attach(m, err);
}

bool vision_service::init_synthetic(model::model & m, const synthetic_rows & rows, std::string & err) {
    release();
    if (m.state_class("vision work (VIS1)") == nullptr || m.state_class("vision staging (VIS1)") == nullptr || !rows) {
        return fail(err, "the vision classes are not reserved, or no row source (the tests' synthetic images)");
    }
    m_     = &m;
    mode_  = mode::synthetic;
    synth_ = rows;
    return own_stream(err) && attach(m, err);
}

bool vision_service::token_row(target & t, bool drafts, std::string & err) {
    const model::reserved_class * wc = m_ == nullptr ? nullptr : m_->state_class("vision weights (VIS1)");
    if (wc == nullptr || wc->plan.parts.size() < 2 || wc->plan.parts[1].bytes[0] == 0) {
        return !drafts || fail(err, "the box drafts over its image rows and the load reserved no image-token row");
    }
    if (wc->plan.parts[1].bytes[0] != model::vision_token_row_bytes ||
        wc->plan.parts[1].bytes[1] != model::vision_token_row_bytes) {
        return fail(err, "the image-token row differs from state.h's layout");
    }
    uint8_t * const      p[2] = { wc->part_ptr(1, 0), wc->part_ptr(1, 1) };
    const int32_t        d[2] = { m_->device[0], m_->device[1] };
    const int32_t        tok  = image_token;
    fn::embed::rows_args a[2];
    const auto           sync = [&](const char * what) {
        for (int32_t k = 0; k < 2; ++k) {
            if (!ok(cudaSetDevice(d[k]), "device", err) || !ok(cudaStreamSynchronize(st(t.stream(k))), what, err)) {
                return false;
            }
        }
        return true;
    };
    for (int32_t k = 0; k < 2; ++k) {
        model::pass_card & pc = t.card(k);
        a[k].table            = pc.token_embd;
        a[k].grids            = pc.c.dense_grids;
        a[k].tok              = reinterpret_cast<const int32_t *>(p[k] + model::vtr_tok);
        a[k].rows             = 1;
        a[k].card             = k;
        a[k].own              = reinterpret_cast<float *>(p[k] + model::vtr_own);
        a[k].full             = reinterpret_cast<float *>(p[k] + model::vtr_full);
        a[k].streams          = reinterpret_cast<float *>(p[k] + model::vtr_streams);
        a[k].err              = reinterpret_cast<uint32_t *>(p[k] + model::vtr_err);
        a[k].q8               = pc.embd_q8;
        if (!ok(cudaSetDevice(d[k]), "device", err) ||
            !ok(cudaMemcpy(p[k] + model::vtr_tok, &tok, 4, cudaMemcpyHostToDevice), "token row", err) ||
            !ok(cudaMemset(p[k] + model::vtr_err, 0, 4), "token row", err)) {
            return false;
        }
        if (!fn::embed::embed_rows(a[k], st(t.stream(k)))) {
            return fail(err, "the image token's embedding (card " + std::to_string(k) + ")");
        }
    }
    if (!sync("token row")) {
        return false;
    }
    for (int32_t k = 0; k < 2; ++k) {
        if (!ok(cudaMemcpyPeer(p[k] + model::vtr_recv, d[k], p[1 - k] + model::vtr_own, d[1 - k],
                               size_t(half_cols) * 4),
                "token row peer", err)) {
            return false;
        }
    }
    for (int32_t k = 0; k < 2; ++k) {
        if (!ok(cudaSetDevice(d[k]), "device", err) ||
            !fn::embed::embed_join(a[k], reinterpret_cast<const float *>(p[k] + model::vtr_recv), st(t.stream(k)))) {
            return err.empty() ? fail(err, "the image token's embedding join") : false;
        }
    }
    if (!sync("token row join")) {
        return false;
    }
    for (int32_t k = 0; k < 2; ++k) {
        uint32_t e = 0;
        if (!ok(cudaSetDevice(d[k]), "device", err) ||
            !ok(cudaMemcpy(&e, p[k] + model::vtr_err, 4, cudaMemcpyDeviceToHost), "token row", err)) {
            return false;
        }
        if (e != 0) {
            return fail(err, "the image token's embedding: error word " + std::to_string(e));
        }
        tok_full_[k] = reinterpret_cast<const float *>(p[k] + model::vtr_full);
    }
    return true;
}

bool vision_service::own_stream(std::string & err) {
    cudaStream_t s = nullptr;
    if (!ok(cudaSetDevice(m_->device[1]), "device", err) ||
        !ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream", err)) {
        release();
        return false;
    }
    own_s1_ = s1_ = s;
    return true;
}

bool vision_service::attach(model::model & m, std::string & err) {
    const model::reserved_class * dc = m.state_class("vision work (VIS1)");
    const model::reserved_class * hc = m.state_class("vision staging (VIS1)");
    lanes_                           = 0;
    for (int32_t l = 0; l < 2; ++l) {
        const int32_t rows = l == 0 ? fn::runtime::pf_chunk_rows : model::lane_rows;
        const int64_t pl   = dc->plan.parts[model::vision_lane_part(model::vp_planes, l)].bytes[0];
        if (pl == 0) {
            break;
        }
        const int64_t sup = dc->plan.parts[model::vision_lane_part(model::vp_supply, l)].bytes[0];
        if (pl != model::vision_plane_bytes(rows) ||
            (mode_ != mode::planes && (sup != int64_t(rows) * half_cols * 4 ||
                                       dc->plan.parts[model::vision_lane_part(model::vp_rows, l)].bytes[1] !=
                                           int64_t(vision::max_tokens) * vision::out_width * 4))) {
            release();
            return fail(err, "the vision work class's lane " + std::to_string(l) + " differs from the plan");
        }
        lane_rows_[l] = rows;
        rows_[l]      = reinterpret_cast<float *>(dc->part_ptr(model::vision_lane_part(model::vp_rows, l), 1));
        for (int32_t c = 0; c < 2; ++c) {
            supply_[l][c] = reinterpret_cast<float *>(dc->part_ptr(model::vision_lane_part(model::vp_supply, l), c));
            planes_[l][c] = dc->part_ptr(model::vision_lane_part(model::vp_planes, l), c);
            slots_[l][c]  = hc->part_ptr(model::vs_planes + l, c);
        }
        ++lanes_;
    }
    if (lanes_ < 1) {
        release();
        return fail(err, "the vision class has no read lane");
    }
    const auto make = [&](void *& e, int32_t c) {
        cudaEvent_t x = nullptr;
        const bool  r = ok(cudaSetDevice(m.device[c]), "device", err) &&
                        ok(cudaEventCreateWithFlags(&x, cudaEventDisableTiming), "event", err);
        e             = x;
        return r;
    };
    bool good = make(ev_blocks_, 0) && make(ev_copied_, 1) && make(ev_supplied_, 1);
    for (int32_t c = 0; c < 2 && good; ++c) {
        good = make(ev_tail_[c], c);
        for (int32_t l = 0; l < lanes_ && good; ++l) {
            good = make(ev_up_[l][c][0], c) && make(ev_up_[l][c][1], c);
        }
    }
    if (!good) {
        const std::string e = err;
        release();
        err = e;
        return false;
    }
    return true;
}

bool vision_service::synthetic(const image_span & s, int32_t lane, std::string & err) {
    // the tests' rows: generated on the host, uploaded on the copy stream and waited for (the host buffer's life)
    std::vector<float> h(size_t(s.tokens()) * vision::out_width);
    synth_(s, h.data());
    if (!ok(cudaSetDevice(m_->device[1]), "device", err) ||
        !ok(cudaMemcpyAsync(rows_[lane], h.data(), h.size() * 4, cudaMemcpyHostToDevice, st(s1_)), "rows", err) ||
        !ok(cudaStreamSynchronize(st(s1_)), "rows", err)) {
        return false;
    }
    return true;
}

bool vision_service::advance(int32_t budget, std::string & err) {
    const queued &           q  = queue_.front();
    const vision::prepared & im = *q.img;
    const int32_t            d0 = m_->device[0], d1 = m_->device[1];
    cudaStream_t             s0 = st(enc_[0].stream()), s1 = st(enc_[1].stream());
    const auto               bad = [&](int32_t c, const char * what) {
        return fail(err, "card " + std::to_string(c) + ": " + (enc_[c].error() != nullptr ? enc_[c].error() : what));
    };
    for (int32_t ran = 0; ran < budget && next_ < vision::blocks;) {
        const int32_t b = next_;
        if (b < vision::vision_split) {
            const int32_t e = std::min(vision::vision_split, b + (budget - ran));
            if (!ok(cudaSetDevice(d0), "device", err)) {
                return false;
            }
            // card 0 overwrites its residual only after card 1 copied the previous one
            if (b == 0 &&
                (!ok(cudaStreamWaitEvent(s0, ev(ev_copied_), 0), "wait copy", err) || !enc_[0].embed(w_[0], im))) {
                return err.empty() ? bad(0, "embed") : false;
            }
            if (!enc_[0].run_blocks(w_[0], b, e)) {
                return bad(0, "encoder");
            }
            if (e == vision::vision_split) {
                if (!ok(cudaEventRecord(ev(ev_blocks_), s0), "record blocks", err) ||
                    !ok(cudaSetDevice(d1), "device", err) ||
                    !ok(cudaStreamWaitEvent(s1, ev(ev_blocks_), 0), "wait blocks", err)) {
                    return false;
                }
                if (!enc_[1].begin_from_residual(enc_[0].residual(), im.gw, im.gh)) {
                    return bad(1, "residual");
                }
                if (!ok(cudaEventRecord(ev(ev_copied_), s1), "record copy", err)) {
                    return false;
                }
            }
            ran += e - b;
            next_ = e;
        } else {
            const int32_t e = std::min(vision::blocks, b + (budget - ran));
            if (!ok(cudaSetDevice(d1), "device", err) || !enc_[1].run_blocks(w_[1], b, e)) {
                return err.empty() ? bad(1, "encoder") : false;
            }
            ran += e - b;
            next_ = e;
        }
    }
    if (next_ == vision::blocks && !enc_[1].merge(w_[1], rows_[q.lane])) {
        return bad(1, "merger");
    }
    return true;
}

bool vision_service::copy_run(int32_t lane, const image_run & u, int64_t done, std::string & err) {
    const int32_t d[2] = { m_->device[0], m_->device[1] };
    cudaStream_t  s1   = st(s1_);
    const float * src  = rows_[lane] + u.k0 * vision::out_width;
    if (!ok(cudaSetDevice(d[1]), "device", err)) {
        return false;
    }
    // card 1's half on its own device; card 0's half across (a pitched peer copy)
    if (!ok(cudaMemcpy2DAsync(supply_[lane][1] + done * half_cols, size_t(half_cols) * 4, src + half_cols,
                              size_t(vision::out_width) * 4, size_t(half_cols) * 4, size_t(u.n),
                              cudaMemcpyDeviceToDevice, s1),
            "supplied rows (card 1)", err)) {
        return false;
    }
    cudaMemcpy3DPeerParms p = {};
    p.srcDevice             = d[1];
    p.srcPtr    = make_cudaPitchedPtr(const_cast<float *>(src), size_t(vision::out_width) * 4, size_t(half_cols) * 4,
                                      size_t(u.n));
    p.dstDevice = d[0];
    p.dstPtr    = make_cudaPitchedPtr(supply_[lane][0] + done * half_cols, size_t(half_cols) * 4, size_t(half_cols) * 4,
                                      size_t(u.n));
    p.extent    = make_cudaExtent(size_t(half_cols) * 4, size_t(u.n), 1);
    return ok(cudaMemcpy3DPeerAsync(&p, s1), "supplied rows (card 0)", err);
}

bool vision_service::supplied(void * const streams[2], std::string & err) {
    if (!ok(cudaSetDevice(m_->device[1]), "device", err) ||
        !ok(cudaEventRecord(ev(ev_supplied_), st(s1_)), "record supplied", err)) {
        return false;
    }
    for (int32_t k = 0; k < 2; ++k) {
        if (!ok(cudaSetDevice(m_->device[k]), "device", err) ||
            !ok(cudaStreamWaitEvent(st(streams[k]), ev(ev_supplied_), 0), "wait supplied", err)) {
            return false;
        }
    }
    return true;
}

bool vision_service::pending(int32_t lane) const {
    for (const queued & q : queue_) {
        if (q.lane == lane) {
            return true;
        }
    }
    return false;
}

bool vision_service::encode_step(int32_t lane, int32_t blocks, std::string & err) {
    int32_t budget = std::max(1, blocks);
    while (budget > 0 && pending(lane)) {
        // the queue's front, whichever lane's (one encoder: images one after another)
        const auto    t0  = std::chrono::steady_clock::now();
        const int32_t was = next_;
        // the tests' synthetic rows: the whole image in one slice
        bool          good =
            mode_ == mode::synthetic ? synthetic(queue_.front().span, queue_.front().lane, err) : advance(budget, err);
        next_ = mode_ == mode::synthetic && good ? vision::blocks : next_;
        budget -= next_ - was;
        // the slice complete before the step returns (a read step is synchronous; writers' rounds follow it)
        for (int32_t k = 0; k < 2 && good && mode_ == mode::encoder; ++k) {
            good = ok(cudaSetDevice(m_->device[k]), "device", err) &&
                   ok(cudaStreamSynchronize(st(enc_[k].stream())), "encoder slice", err);
        }
        if (!good) {
            forget(queue_.front().lane);
            return false;
        }
        queued & q = queue_.front();
        q.wall_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++q.steps;
        if (next_ < vision::blocks) {
            break;  // the budget spent inside the image
        }
        // merged: the lane holds the image; its run copied to the lane's supply
        held_[q.lane] = { q.slot, q.span.cell, q.span.digest, q.span.nx, q.span.ny };
        ++encodes_;
        std::fprintf(stderr,
                     "FNFL vision: slot %d: the image at cell %lld (%d x %d tokens) encoded in %d step%s, %.1f ms "
                     "(measured: its slices' host wall time)\n",
                     q.slot, static_cast<long long>(q.span.cell), q.span.nx, q.span.ny, q.steps, q.steps > 1 ? "s" : "",
                     q.wall_s * 1e3);
        const queued done = queue_.front();
        queue_.pop_front();
        next_ = 0;
        if (!copy_run(done.lane, done.run, done.done, err) || (!pending(done.lane) && !supplied(done.streams, err)) ||
            !ok(cudaSetDevice(m_->device[1]), "device", err) ||
            !ok(cudaStreamSynchronize(st(s1_)), "supplied rows", err)) {
            forget(done.lane);
            return false;
        }
    }
    return true;
}

void vision_service::forget(int32_t lane) {
    if (!queue_.empty() && queue_.front().lane == lane && next_ > 0) {
        // the front's encode stopped at its last slice (each slice ends synchronized; drained again here)
        for (int32_t k = 0; k < 2; ++k) {
            cudaSetDevice(m_->device[k]);
            cudaStreamSynchronize(st(enc_[k].stream()));
        }
        next_ = 0;
    }
    for (auto it = queue_.begin(); it != queue_.end();) {
        it = it->lane == lane ? queue_.erase(it) : it + 1;
    }
    if (lane >= 0 && lane < 2) {
        held_[lane] = held();
    }
}

bool vision_service::fill(target &              t,
                          int32_t               lane,
                          const image_chunk &   h,
                          model::prompt_chunk & c,
                          bool                  sliced,
                          std::string &         err) {
    void * const s[2] = { t.stream(0), t.stream(1) };
    return fill_rows(
        t.slot_index(), t.images(), [&](size_t i) { return t.image_pixels(i); }, s, lane, h, c, sliced, err);
}

bool vision_service::fill_rows(int32_t               slot,
                               const image_rows &    rows,
                               const pixels &        pix,
                               void * const          streams[2],
                               int32_t               lane,
                               const image_chunk &   h,
                               model::prompt_chunk & c,
                               bool                  sliced,
                               std::string &         err) {
    if (m_ == nullptr || lane < 0 || lane >= lanes_ || c.rows < 1 || c.rows > lane_rows_[lane] ||
        int64_t(h.planes.size()) != int64_t(c.rows) * 4) {
        return fail(err, "a chunk of " + std::to_string(c.rows) + " rows in read lane " + std::to_string(lane));
    }
    if (pending(lane)) {
        return fail(err, "read lane " + std::to_string(lane) + " still waits for an image of its previous chunk");
    }
    if (!queue_rows(slot, rows, pix, streams, lane, h, c, err)) {
        forget(lane);
        return false;
    }
    while (!sliced && pending(lane)) {
        if (!encode_step(lane, vision::blocks, err)) {
            return false;
        }
    }
    return true;
}

bool vision_service::queue_rows(int32_t               slot,
                                const image_rows &    rows,
                                const pixels &        pix,
                                void * const          streams[2],
                                int32_t               lane,
                                const image_chunk &   h,
                                model::prompt_chunk & c,
                                std::string &         err) {
    const int32_t                   d[2] = { m_->device[0], m_->device[1] };
    const std::vector<image_span> & sp   = rows.spans();
    cudaStream_t                    s1   = st(s1_);
    runs_[lane].clear();
    int64_t done       = 0;
    bool    tails      = false;
    bool    queued_any = false;
    // the split's back embeds nothing: its chunks take the planes only
    for (const image_run & u : mode_ == mode::planes ? std::vector<image_run>() : h.runs) {
        if (u.image < 0 || size_t(u.image) >= sp.size()) {
            return fail(err, "a run of image " + std::to_string(u.image));
        }
        const image_span & s  = sp[size_t(u.image)];
        held &             hd = held_[lane];
        if (!tails) {
            // the lane's previous chunk read its supplied rows from the target streams: copy after them
            for (int32_t k = 0; k < 2; ++k) {
                if (!ok(cudaSetDevice(d[k]), "device", err) ||
                    !ok(cudaEventRecord(ev(ev_tail_[k]), st(streams[k])), "record tail", err)) {
                    return false;
                }
            }
            if (!ok(cudaSetDevice(d[1]), "device", err) ||
                !ok(cudaStreamWaitEvent(s1, ev(ev_tail_[0]), 0), "wait tail", err) ||
                !ok(cudaStreamWaitEvent(s1, ev(ev_tail_[1]), 0), "wait tail", err)) {
                return false;
            }
            tails = true;
        }
        bool now = hd.slot == slot && hd.cell == s.cell && hd.digest == s.digest && hd.nx == s.nx && hd.ny == s.ny;
        if (!now) {
            hd                                          = held();
            // the tests' synthetic rows need no pixels (encode_step generates them)
            std::shared_ptr<const vision::prepared> img = mode_ == mode::synthetic ? nullptr : pix(size_t(u.image));
            if (mode_ != mode::synthetic && img == nullptr) {
                return fail(
                    err, "the image at cell " + std::to_string(s.cell) + " has no prepared image (its rows are read)");
            }
            if (img != nullptr && (img->nx != s.nx || img->ny != s.ny || img->digest != s.digest)) {
                return fail(err, "the image at cell " + std::to_string(s.cell) + " differs from its span");
            }
            queued q;
            q.lane       = lane;
            q.slot       = slot;
            q.span       = s;
            q.img        = std::move(img);
            q.run        = u;
            q.done       = done;
            q.streams[0] = streams[0];
            q.streams[1] = streams[1];
            queue_.push_back(std::move(q));
            queued_any = true;
        }
        // a held image's rows now (stream order before a later image's merger into the lane's rows)
        if (now && !copy_run(lane, u, done, err)) {
            return false;
        }
        runs_[lane].push_back({ u.row0, u.n });
        done += u.n;
    }
    if (!runs_[lane].empty() && !queued_any && !supplied(streams, err)) {
        return false;
    }
    // the planes: a pinned slot (its previous upload complete), then each card's copy on its target stream
    const int32_t sl     = turn_[lane];
    turn_[lane]          = 1 - sl;
    const int64_t pbytes = model::vision_plane_bytes(lane_rows_[lane]);
    const int64_t boff   = model::vision_bplane_offset(lane_rows_[lane]);
    const size_t  nb     = h.bplanes.size() * 4;
    if (int64_t(c.rows) * 16 > boff || boff + int64_t(nb) > pbytes) {
        return fail(err, "the planes of a chunk of " + std::to_string(c.rows) + " rows");
    }
    for (int32_t k = 0; k < 2; ++k) {
        uint8_t *    host = slots_[lane][k] + sl * pbytes;
        cudaStream_t ts   = st(streams[k]);
        if (!ok(cudaSetDevice(d[k]), "device", err) ||
            !ok(cudaEventSynchronize(ev(ev_up_[lane][k][sl])), "plane slot", err)) {
            return false;
        }
        std::memcpy(host, h.planes.data(), h.planes.size() * 4);
        std::memcpy(host + boff, h.bplanes.data(), nb);
        const bool up =
            ok(cudaMemcpyAsync(planes_[lane][k], host, h.planes.size() * 4, cudaMemcpyHostToDevice, ts), "planes",
               err) &&
            (nb == 0 ||
             ok(cudaMemcpyAsync(planes_[lane][k] + boff, host + boff, nb, cudaMemcpyHostToDevice, ts), "planes", err));
        if (!up || !ok(cudaEventRecord(ev(ev_up_[lane][k][sl]), ts), "record planes", err)) {
            return false;
        }
        c.planes[k]  = reinterpret_cast<const int32_t *>(planes_[lane][k]);
        c.bplanes[k] = reinterpret_cast<const int32_t *>(planes_[lane][k] + boff);
        c.embd[k]    = runs_[lane].empty() ? nullptr : supply_[lane][k];
        c.draft_e[k] = runs_[lane].empty() ? nullptr : tok_full_[k];
    }
    c.runs   = runs_[lane].empty() ? nullptr : runs_[lane].data();
    c.n_runs = static_cast<int32_t>(runs_[lane].size());
    return true;
}

}  // namespace fn::runtime
