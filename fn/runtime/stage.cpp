#include "fn/runtime/stage.h"

#include "fn/kernels/head/head.h"
#include "fn/model/pass.h"
#include "fn/model/prompt.h"
#include "fn/runtime/joint.h"
#include "fn/runtime/nvtx.h"
#include "fn/xchg/life.h"
#include "fn/xchg/slow.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>

namespace fn::runtime::stage {

namespace {

// VIS1 Stage B: a read command's payload: its tokens, then each image's span in span_words words (cell low and high,
// nx, ny, digest low and high; the follower reads them without pixels: its back takes the planes only). ctrl::b is the
// image count, ctrl::n_tokens the payload's words
constexpr int32_t span_words = 6;

std::vector<int32_t> read_payload(const int32_t *                   tokens,
                                  int32_t                           n_tokens,
                                  const std::vector<prompt_image> * images,
                                  int32_t &                         n_images) {
    std::vector<int32_t> p(tokens, tokens + (n_tokens > 0 ? n_tokens : 0));
    n_images = images != nullptr ? static_cast<int32_t>(images->size()) : 0;
    for (int32_t i = 0; i < n_images; ++i) {
        const image_span s             = span_of((*images)[size_t(i)]);
        const int32_t    w[span_words] = {
            int32_t(uint64_t(s.cell) & 0xffffffffu), int32_t(uint64_t(s.cell) >> 32), s.nx, s.ny,
            int32_t(s.digest & 0xffffffffu),         int32_t(s.digest >> 32)
        };
        p.insert(p.end(), w, w + span_words);
    }
    return p;
}

// the follower's side: the tokens' count and the images (spans only); false when the words do not add up
bool payload_images(const std::vector<int32_t> & p,
                    int32_t                      words,
                    int32_t                      n_images,
                    int32_t &                    n_tokens,
                    std::vector<prompt_image> &  out) {
    out.clear();
    n_tokens = words - span_words * n_images;
    if (n_images < 0 || n_tokens < 0 || int64_t(p.size()) < words) {
        return false;
    }
    for (int32_t i = 0; i < n_images; ++i) {
        const int32_t * w = p.data() + n_tokens + span_words * i;
        prompt_image    pi;
        pi.span.cell   = int64_t(uint64_t(uint32_t(w[0])) | (uint64_t(uint32_t(w[1])) << 32));
        pi.span.nx     = w[2];
        pi.span.ny     = w[3];
        pi.span.digest = uint64_t(uint32_t(w[4])) | (uint64_t(uint32_t(w[5])) << 32);
        out.push_back(pi);
    }
    return true;
}

namespace lk = fn::xchg::link;
using clk    = std::chrono::steady_clock;

constexpr double  ctrl_timeout_s = 60.0;   // a command's answer (a prefill's: crossing timeout)
constexpr double  step_timeout_s = 120.0;  // a step group's X-STEP (back half, catch-up, next draft)
constexpr int32_t max_b          = fn::round::max_rows;

double since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

bool fail(std::string & err, const std::string & what) {
    err = what;
    return false;
}

// SSF1: a follower command (op, slot, group; its read groups included) or read group of slow::threshold_s or more
// while writers generate on this box: a trace line at the scope's end
struct slow_scope {
    const char *    what;
    uint32_t        seq = 0;
    int32_t         op = -1, slot = -1, group = -1;
    clk::time_point t0 = clk::now();

    ~slow_scope() {
        const double s = since(t0);
        if (fn::xchg::slow::due(s)) {
            std::string d = "slot " + std::to_string(slot);
            if (op >= 0) {
                d = "seq " + std::to_string(seq) + " op " + std::to_string(op) + " " + d + " group " +
                    std::to_string(group);
            }
            fn::xchg::slow::line(what, s, d);
        }
    }
};

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e != cudaSuccess) {
        err = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

void set_why(char * dst, size_t cap, const std::string & s) {
    std::memset(dst, 0, cap);
    std::memcpy(dst, s.data(), std::min(cap - 1, s.size()));
}

std::string why_of(const char * p, size_t cap) {
    return std::string(p, strnlen(p, cap));
}

// APF1: a box's pacing totals in a read's answer (ack: acct[5..7], acct_i[10..21]) and back
int32_t clamp32(int64_t v) {
    return static_cast<int32_t>(std::clamp<int64_t>(v, INT32_MIN, INT32_MAX));
}

void pace_to_ack(const pace::stats & p, ack & a) {
    a.acct[5]    = p.longest_ms;
    a.acct[6]    = p.over_ms_sum;
    a.acct[7]    = p.budget_sum;
    a.acct_i[10] = clamp32(p.steps);
    a.acct_i[11] = clamp32(p.known);
    a.acct_i[12] = clamp32(p.zero);
    a.acct_i[13] = clamp32(p.unknown);
    a.acct_i[14] = clamp32(p.completions);
    a.acct_i[15] = clamp32(p.gaps);
    a.acct_i[16] = clamp32(p.over);
    a.acct_i[17] = clamp32(p.over_none);
    a.acct_i[18] = clamp32(p.over_budgeted);
    a.acct_i[19] = clamp32(p.over_minimum);
    a.acct_i[20] = clamp32(p.over_small);
    a.acct_i[21] = clamp32(p.over_idle);
}

void pace_from_ack(const ack & a, pace::stats & p) {
    p.longest_ms    = a.acct[5];
    p.over_ms_sum   = a.acct[6];
    p.budget_sum    = a.acct[7];
    p.steps         = a.acct_i[10];
    p.known         = a.acct_i[11];
    p.zero          = a.acct_i[12];
    p.unknown       = a.acct_i[13];
    p.completions   = a.acct_i[14];
    p.gaps          = a.acct_i[15];
    p.over          = a.acct_i[16];
    p.over_none     = a.acct_i[17];
    p.over_budgeted = a.acct_i[18];
    p.over_minimum  = a.acct_i[19];
    p.over_small    = a.acct_i[20];
    p.over_idle     = a.acct_i[21];
}

bool is_eog(const std::vector<int32_t> & eog, int32_t tok) {
    return std::find(eog.begin(), eog.end(), tok) != eog.end();
}

// the step_info of a member from its X-STEP (the leader's record of the step the back ran)
step_info info_of(const step_out & o) {
    step_info s;
    s.kind     = o.kind;
    s.n_before = o.n_before;
    s.d        = o.d;
    s.a        = o.a;
    s.n_out    = o.n_out;
    for (int32_t j = 0; j < 5; ++j) {
        s.out[j] = j < o.n_out ? o.out[j] : -1;
    }
    s.end          = o.end != 0;
    s.cut          = o.cut != 0;
    s.waits        = o.waits;
    s.wall_s       = o.wall_s;
    s.to_draft_s   = o.to_draft_s;
    s.to_verify_s  = o.to_verify_s;
    s.to_catchup_s = o.to_catchup_s;
    s.to_one_s     = o.to_one_s;
    s.joint        = o.joint != 0;
    s.hash         = o.hash;
    return s;
}

}  // namespace

// ---- the boundary buffers ----

buffers::~buffers() {
    // the link stopped first (its owner's teardown): no operation touches these
    for (auto & s : slot) {
        for (uint8_t * p : s) {
            if (p != nullptr) {
                life::disown(p, "boundary buffer");
                cudaFreeHost(p);
            }
        }
    }
}

bool buffers::alloc(int32_t n_slots, port * p, std::string & err) {
    slot.assign(size_t(n_slots), { nullptr, nullptr });
    const size_t n = size_t(model::stage_bytes(max_b));
    for (int32_t s = 0; s < n_slots; ++s) {
        for (int32_t c = 0; c < 2; ++c) {
            void * q = nullptr;
            if (!cuda_ok(cudaHostAlloc(&q, n, cudaHostAllocPortable), "boundary buffer", err)) {
                return false;
            }
            std::memset(q, 0, n);
            life::own(q, n, "boundary buffer");
            slot[size_t(s)][size_t(c)] = static_cast<uint8_t *>(q);
            if (p != nullptr && !p->map(q, n, err)) {
                return false;
            }
        }
    }
    return true;
}

// ---- the prompt crossing ----

static_assert(sizeof(crossing_head::cut) / sizeof(int32_t) == size_t(model::prompt_max_pieces), "the head's partition");

bool crossing::init(port & p, bool front, int32_t k, std::string & err) {
    p_     = &p;
    front_ = front;
    k_     = k;
    model::prompt_chunk big;
    big.rows    = model::prompt_chunk_rows;
    big.half[0] = model::prompt_half_rows;
    big.half[1] = model::prompt_chunk_rows - model::prompt_half_rows;
    // MUP2: the most pieces (every piece's FFN messages carry a header): 128 of 64 rows
    std::string e2;
    if (!model::prompt_set_pieces(big, model::prompt_chunk_rows / model::prompt_max_pieces, e2)) {
        return fail(err, "crossing: " + e2);
    }
    cap_ = model::prompt_path::crossing_bytes(big);
    for (int32_t b = 0; b < 2; ++b) {
        void * q = nullptr;
        if (!cuda_ok(cudaHostAlloc(&q, size_t(cap_), cudaHostAllocPortable), "crossing buffer", err)) {
            return false;
        }
        life::own(q, size_t(cap_), "crossing buffer");
        buf_[b] = static_cast<uint8_t *>(q);
        if (!p.map(q, size_t(cap_), err)) {
            return false;
        }
    }
    return true;
}

crossing::~crossing() {
    for (int32_t b = 0; b < 2; ++b) {
        if (buf_[b] != nullptr) {
            life::disown(buf_[b], "crossing buffer");
            cudaFreeHost(buf_[b]);
        }
        for (int32_t c = 0; c < 2; ++c) {
            if (done_[b][c] != nullptr) {
                cudaEventDestroy(static_cast<cudaEvent_t>(done_[b][c]));
            }
        }
    }
}

void crossing::expect(int32_t slot, int64_t base) {
    if (slot >= 0 && slot < max_split_slots) {
        retire_ahead(slot);  // PRD1 item 11: a new read: nothing of the last one stays taken ahead
        base_[slot]       = base;
        sent_[slot]       = 0;
        taken_[slot]      = base;
        plan_[slot]       = 0;
        ahead_done_[slot] = 0;
    }
}

void crossing::plan(int32_t slot, int64_t chunks) {
    if (slot >= 0 && slot < max_split_slots) {
        plan_[slot] = chunks;
    }
}

void crossing::retire_ahead(int32_t slot) {
    if (ahead_.slot < 0 || (slot >= 0 && ahead_.slot != slot)) {
        return;
    }
    std::vector<int32_t> ops = ahead_.ops;
    if (!ahead_.head && ahead_.hop >= 0) {
        ops.push_back(ahead_.hop);
    }
    if (!ops.empty()) {
        p_->retire(ops.data(), int32_t(ops.size()));
    }
    ahead_ = ahead_rx();
}

bool crossing::buffer_free(int32_t b, bool wait, std::string & err) {
    for (int32_t q = 0; q < 2 && used_[b]; ++q) {
        if (!cuda_ok(cudaSetDevice(dev_[q]), "device", err)) {
            return false;
        }
        if (wait) {
            if (!cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(done_[b][q])), "crossing buffer", err)) {
                return false;
            }
        } else {
            const cudaError_t e = cudaEventQuery(static_cast<cudaEvent_t>(done_[b][q]));
            if (e == cudaErrorNotReady) {
                return false;
            }
            if (!cuda_ok(e, "crossing buffer", err)) {
                return false;
            }
        }
    }
    return true;
}

bool crossing::post_pieces(const crossing_head &  h,
                           int64_t                n,
                           int32_t                b,
                           std::vector<int32_t> & ops,
                           std::string &          err) {
    life::reuse(buf_[b], size_t(cap_), "crossing buffer");
    for (int32_t i = 0; i < h.pieces; ++i) {
        const int64_t at = int64_t(i) * piece_bytes;
        const int32_t op =
            p_->recv(crossing_tag(n, i), buf_[b] + at, size_t(std::min<int64_t>(piece_bytes, h.bytes - at)), err);
        if (op < 0) {
            p_->retire(ops.data(), int32_t(ops.size()));
            ops.clear();
            return false;
        }
        ops.push_back(op);
    }
    return true;
}

bool crossing::poll(bool next, std::string & err) {
    if (front_ || ahead_.slot < 0 || ahead_.posted) {
        return true;
    }
    if (!ahead_.head) {
        bool done = false;
        if (!p_->test(ahead_.hop, done, err)) {
            ahead_ = ahead_rx();  // the receive failed (finished): in takes the header afresh and fails there
            return false;
        }
        ahead_.head = done;
    }
    const crossing_head & h = *ahead_.hr;
    const int32_t         b = static_cast<int32_t>(ahead_.n & 1);
    // a header in() will refuse is left to it (its pieces into a scratch buffer)
    const bool sane         = h.bytes > 0 && h.bytes <= cap_ && h.pieces == (h.bytes + piece_bytes - 1) / piece_bytes;
    if (!ahead_.head || !next || !sane || dev_[0] < 0) {
        return true;
    }
    std::string e2;
    if (!buffer_free(b, false, e2)) {
        if (!e2.empty()) {
            err = e2;
            return false;
        }
        return true;  // its previous copy-in still runs: a later poll
    }
    if (!post_pieces(h, ahead_.n, b, ahead_.ops, err)) {
        return false;
    }
    ahead_.posted = true;
    return true;
}

bool crossing::drop(int32_t slot, int64_t base, int64_t count, std::string & err) {
    if (front_ || slot < 0 || slot >= max_split_slots) {
        return fail(err, "crossing drop: the follower's, a slot");
    }
    for (int64_t n = std::max(taken_[slot], base); n < base + count; ++n) {
        // PRD1 item 11: a crossing taken ahead is finished through its own receives (never received twice)
        if (ahead_.slot == slot && ahead_.n == n) {
            ahead_rx a = ahead_;
            ahead_     = ahead_rx();
            if ((!a.head && !p_->wait(a.hop, err, timeout_s)) || (a.posted && !wait_all(*p_, a.ops, timeout_s, err))) {
                return false;
            }
            if (a.posted) {
                taken_[slot] = n + 1;
                continue;
            }
            auto tmp = std::make_shared<std::vector<uint8_t>>(size_t(std::clamp<int64_t>(a.hr->bytes, 0, piece_bytes)));
            for (int32_t i = 0; i < a.hr->pieces; ++i) {
                const int64_t at = int64_t(i) * piece_bytes;
                const int32_t op = p_->recv(crossing_tag(n, i), tmp->data(),
                                            size_t(std::clamp<int64_t>(a.hr->bytes - at, 0, piece_bytes)), tmp, err);
                if (op < 0 || !p_->wait(op, err, timeout_s)) {
                    return false;
                }
            }
            taken_[slot] = n + 1;
            continue;
        }
        // the records on the heap, kept by each receive: a wait given up never leaves one writing a freed frame
        auto          h   = std::make_shared<crossing_head>();
        const int32_t hop = p_->recv(crossing_tag(n, 255), h.get(), sizeof(*h), h, err);
        if (hop < 0 || !p_->wait(hop, err, timeout_s)) {
            return false;
        }
        auto tmp = std::make_shared<std::vector<uint8_t>>(size_t(std::clamp<int64_t>(h->bytes, 0, piece_bytes)));
        for (int32_t i = 0; i < h->pieces; ++i) {
            const int64_t at = int64_t(i) * piece_bytes;
            const int32_t op = p_->recv(crossing_tag(n, i), tmp->data(),
                                        size_t(std::clamp<int64_t>(h->bytes - at, 0, piece_bytes)), tmp, err);
            if (op < 0 || !p_->wait(op, err, timeout_s)) {
                return false;
            }
        }
        taken_[slot] = n + 1;
    }
    retire_ahead(slot);  // PRD1 item 11: a chunk past the sent ones is never sent
    return true;
}

bool crossing::out(target & t, model::prompt_path & pp, const model::prompt_chunk & c, std::string & err) {
    if (!front_) {
        return fail(err, "crossing: the follower sends none");
    }
    const int32_t slot  = t.slot_index();
    const int64_t n     = base_[slot] + c.index;
    const int64_t bytes = model::prompt_path::crossing_bytes(c);
    if (abort && abort()) {
        return fail(err, "crossing: the follower failed this read (its answer came first)");
    }
    if (bytes > cap_) {
        return fail(err, "crossing: " + std::to_string(bytes) + " bytes beyond the buffer's " + std::to_string(cap_));
    }
    const int32_t b = static_cast<int32_t>(n & 1);
    // release: the buffer's previous sends ended (a failure retires every send of both buffers first)
    if (!wait_all(*p_, ops_[b], timeout_s, err)) {
        retire_sends();
        return false;
    }
    ops_[b].clear();
    life::reuse(buf_[b], size_t(cap_), "crossing buffer");
    if (!pp.crossing_out(c, k_, buf_[b], err)) {
        return false;
    }
    for (int32_t q = 0; q < 2; ++q) {
        if (!cuda_ok(cudaSetDevice(t.card(q).c.device), "device", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(q))), "crossing out", err)) {
            return false;
        }
    }
    // the header's record on the heap, kept by its send
    auto            hr = std::make_shared<crossing_head>();
    crossing_head & h  = *hr;
    h.slot            = slot;
    h.rows            = c.rows;
    h.half[0]         = c.half[0];
    h.half[1]         = c.half[1];
    h.start           = c.start;
    h.seq             = c.seq;
    h.bytes           = bytes;
    h.pieces          = static_cast<int32_t>((bytes + piece_bytes - 1) / piece_bytes);
    h.n_cuts           = c.n_pieces;
    for (int32_t j = 0; j < c.n_pieces; ++j) {
        h.cut[j] = c.piece[j];
    }
    // the header first, waited, then the pieces from the pinned buffer (its sends waited before its next refill)
    const int32_t hop  = p_->send(crossing_tag(n, 255), hr.get(), sizeof(h), hr, err);
    if (hop < 0 || !p_->wait(hop, err, timeout_s)) {
        retire_sends();
        return false;
    }
    for (int32_t i = 0; i < h.pieces; ++i) {
        const int64_t at = int64_t(i) * piece_bytes;
        const int32_t op =
            p_->send(crossing_tag(n, i), buf_[b] + at, size_t(std::min<int64_t>(piece_bytes, bytes - at)), err);
        if (op < 0) {
            retire_sends();
            return false;
        }
        ops_[b].push_back(op);
    }
    owner_[b]   = slot;
    sent_[slot] = std::max<int64_t>(sent_[slot], c.index + 1);
    ++n_;
    bytes_ += bytes;
    return true;
}

bool crossing::drain(std::string & err) {
    return drain(-1, err);
}

bool crossing::drain(int32_t slot, std::string & err) {
    for (int32_t b = 0; b < 2; ++b) {
        if (slot >= 0 && owner_[b] != slot) {
            continue;
        }
        if (!wait_all(*p_, ops_[b], timeout_s, err)) {
            retire_sends();
            return false;
        }
        ops_[b].clear();
    }
    return true;
}

void crossing::retire_sends() {
    for (int32_t b = 0; b < 2; ++b) {
        if (!ops_[b].empty()) {
            p_->retire(ops_[b].data(), int32_t(ops_[b].size()));
        }
        ops_[b].clear();
    }
}

bool crossing::in(target & t, model::prompt_path & pp, model::prompt_chunk & c, std::string & err) {
    if (front_) {
        return fail(err, "crossing: the leader receives none");
    }
    const int32_t slot = t.slot_index();
    const int64_t n    = base_[slot] + c.index;
    const int32_t b    = static_cast<int32_t>(n & 1);
    for (int32_t q = 0; q < 2; ++q) {
        dev_[q] = t.card(q).c.device;
    }
    // PRD1 item 11: this chunk taken ahead (its header receive posted at the last chunk's copy-in, its pieces maybe
    // posted by poll), or another of this slot's: retired (never taken twice)
    std::shared_ptr<crossing_head> hr;
    std::vector<int32_t>           ops;
    bool                           posted = false, landed = false;
    if (ahead_.slot == slot && ahead_.n == n) {
        hr     = ahead_.hr;
        posted = ahead_.posted;
        // the pieces still out at entry (a completed test finishes its op)
        for (const int32_t op : ahead_.ops) {
            bool done = false;
            if (!p_->test(op, done, err)) {
                p_->retire(ahead_.ops.data(), int32_t(ahead_.ops.size()));
                ahead_ = ahead_rx();
                return false;
            }
            if (!done) {
                ops.push_back(op);
            }
        }
        landed = posted && ops.empty();
        if (!ahead_.head && !p_->wait(ahead_.hop, err, timeout_s)) {
            p_->retire(ops.data(), int32_t(ops.size()));
            ahead_ = ahead_rx();
            return false;
        }
        ahead_ = ahead_rx();
    } else {
        retire_ahead(slot);
        hr                = std::make_shared<crossing_head>();  // kept by its receive
        const int32_t hop = p_->recv(crossing_tag(n, 255), hr.get(), sizeof(crossing_head), hr, err);
        if (hop < 0 || !p_->wait(hop, err, timeout_s)) {
            return false;
        }
    }
    crossing_head & h  = *hr;
    taken_[slot]       = n;  // the head is taken: a mismatch below still takes the pieces, then fails
    // MUP2: the front's partition (validated with the rest)
    const bool cuts_ok = h.n_cuts >= 0 && h.n_cuts <= model::prompt_max_pieces;
    if (cuts_ok) {
        c.n_pieces = h.n_cuts;
        for (int32_t j = 0; j < model::prompt_max_pieces; ++j) {
            c.piece[j] = j < h.n_cuts ? h.cut[j] : 0;
        }
    }
    const bool match = cuts_ok && model::prompt_pieces_ok(c) && h.slot == slot && h.start == c.start &&
                       h.rows == c.rows && h.half[0] == c.half[0] && h.half[1] == c.half[1] &&
                       h.bytes == model::prompt_path::crossing_bytes(c) && h.bytes <= cap_;
    if (!match) {
        if (posted) {
            std::string e2;
            wait_all(*p_, ops, timeout_s, e2);  // its pieces, posted ahead into the free buffer, end first
        } else {
            auto tmp = std::make_shared<std::vector<uint8_t>>(size_t(std::clamp<int64_t>(h.bytes, 0, piece_bytes)));
            for (int32_t i = 0; i < h.pieces && h.bytes > 0; ++i) {
                const int64_t at = int64_t(i) * piece_bytes;
                std::string   e2;
                const int32_t op = p_->recv(crossing_tag(n, i), tmp->data(),
                                            size_t(std::clamp<int64_t>(h.bytes - at, 0, piece_bytes)), tmp, e2);
                if (op < 0 || !p_->wait(op, e2, timeout_s)) {
                    break;
                }
            }
        }
        taken_[slot] = n + 1;
        return fail(err, "crossing " + std::to_string(n) + ": the leader's chunk (slot " + std::to_string(h.slot) +
                             ", " + std::to_string(h.start) + "+" + std::to_string(h.rows) +
                             ") is not this one (slot " + std::to_string(slot) + ", " + std::to_string(c.start) + "+" +
                             std::to_string(c.rows) + ")");
    }
    if (!posted) {
        // the buffer's previous copy-in completed on both cards, then it is refilled; every piece's receive is
        // waited or retired
        if (!buffer_free(b, true, err) || !post_pieces(h, n, b, ops, err)) {
            return false;
        }
    }
    // PRD1 item 11: the copy-in starts: the next chunk's header received ahead, when it is in the read's plan
    if (c.index + 1 < plan_[slot] && ahead_.slot < 0) {
        ahead_.slot = slot;
        ahead_.n    = n + 1;
        ahead_.hr   = std::make_shared<crossing_head>();
        ahead_.hop  = p_->recv(crossing_tag(n + 1, 255), ahead_.hr.get(), sizeof(crossing_head), ahead_.hr, err);
        if (ahead_.hop < 0) {
            ahead_ = ahead_rx();
            p_->retire(ops.data(), int32_t(ops.size()));
            return false;
        }
    }
    if (!wait_all(*p_, ops, timeout_s, err)) {
        return false;
    }
    ahead_done_[slot] += landed ? 1 : 0;
    c.seq = h.seq;  // the front's chunk number: the messages' generations
    if (!pp.crossing_in(c, k_, buf_[b], err)) {
        return false;
    }
    for (int32_t q = 0; q < 2; ++q) {
        if (!cuda_ok(cudaSetDevice(t.card(q).c.device), "device", err)) {
            return false;
        }
        if (done_[b][q] == nullptr) {
            cudaEvent_t e = nullptr;
            if (!cuda_ok(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "crossing event", err)) {
                return false;
            }
            done_[b][q] = e;
        }
        if (!cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(done_[b][q]), static_cast<cudaStream_t>(t.stream(q))),
                     "crossing event", err)) {
            return false;
        }
    }
    used_[b]     = true;
    taken_[slot] = n + 1;
    ++n_;
    bytes_ += h.bytes;
    return true;
}

// ---- X-HELLO's identities ----

namespace {

struct fnv128 {
    uint64_t a = 0xcbf29ce484222325ull, b = 0x84222325cbf29ce4ull;

    void add(const void * p, size_t n) {
        const auto * c = static_cast<const uint8_t *>(p);
        for (size_t i = 0; i < n; ++i) {
            a = (a ^ c[i]) * 0x100000001b3ull;
            b = (b ^ c[i] ^ 0x5a) * 0x100000001b3ull;
        }
    }

    std::string hex() const {
        char s[33];
        snprintf(s, sizeof s, "%016llx%016llx", (unsigned long long) a, (unsigned long long) b);
        return s;
    }
};

}  // namespace

std::string hello_files(const model::model & m) {
    fnv128 h;
    for (const auto & f : m.files) {
        const uint64_t sz = f->size();
        const uint64_t n  = std::min<uint64_t>(sz, uint64_t(16) << 20);
        h.add(&sz, sizeof sz);
        h.add(f->data(), size_t(n));
    }
    return h.hex();
}

std::string hello_settings(int32_t                      k,
                           int32_t                      n_slots,
                           int32_t                      cells,
                           int32_t                      pool_cells,
                           bool                         kv_q8,
                           int32_t                      spec_d,
                           const std::string &          attention,
                           const std::vector<int32_t> & eog,
                           bool                         images) {
    std::string s = "k=" + std::to_string(k) + " slots=" + std::to_string(n_slots) + " cells=" + std::to_string(cells) +
                    " pool=" + std::to_string(pool_cells) + " kv=" + (kv_q8 ? "q8" : "bf16") +
                    " d=" + std::to_string(spec_d) + " att=" + attention + " eog=";
    std::vector<int32_t> e = eog;
    std::sort(e.begin(), e.end());
    for (size_t i = 0; i < e.size(); ++i) {
        s += (i ? "," : "") + std::to_string(e[i]);
    }
    return images ? s + " images=1" : s;  // VIS1: text-only settings keep their text and hash
}

std::string hello_hash(const std::string & text) {
    fnv128 h;
    h.add(text.data(), text.size());
    return h.hex();
}

// ---- the leader ----

struct leader::pending {
    int32_t                     group = 0;
    int32_t                     n     = 0;
    member                      m[max_members];
    int32_t                     b[max_members]  = {};
    run_status                  st[max_members] = {};
    std::string                 why[max_members];
    target::front_round         fr[max_members];
    bool                        fronted[max_members] = {};
    clk::time_point             t0;
    // the wire records (the command, the answers, forced logits), shared with the transport (fn/runtime/stage_wire.h)
    std::shared_ptr<group_wire> w = std::make_shared<group_wire>();
};

bool leader::init(engine & e, port & p, crossing & x, std::string & err) {
    e_ = &e;
    p_ = &p;
    x_ = &x;
    ctl_.init(p);
    if (e.config().stage != 1 || int32_t(e.config().stage_buf.size()) < e.n_slots()) {
        return fail(err, "leader: the engine is not a front stage with its boundary buffers");
    }
    next_.assign(size_t(e.n_slots()), next_step());
    samp_.assign(size_t(e.n_slots()), nullptr);
    read_acks_.assign(size_t(e.n_slots()), {});
    read_base_.assign(size_t(e.n_slots()), 0);
    read_end_.assign(size_t(e.n_slots()), -1);
    back_acct_.assign(size_t(e.n_slots()), read_info());
    logits_.assign(size_t(e.n_slots()), {});
    return true;
}

void leader::drain_read(int32_t slot, int64_t base) {
    if (ctl_.lost()) {
        return;
    }
    std::string e;
    if (!x_->drain(e)) {
        ctl_.lose(e);
        return;
    }
    ctrl c;
    c.op   = int32_t(op::drain);
    c.slot = slot;
    c.base = base;
    c.a    = static_cast<int32_t>(x_->sent(slot));
    ack a;
    if (command(c, nullptr, a, 3600.0) && a.status != 0) {
        ctl_.lose("the follower could not drop a failed read's crossings: " + why_of(a.why, sizeof a.why));
    }
}

bool leader::command(ctrl & c, const int32_t * tokens, ack & a, double timeout_s) {
    const double w0 = ctl_.wait_s;
    const bool   ok = ctl_.command(c, tokens, a, timeout_s);
    st_.back_wait_s += ctl_.wait_s - w0;
    if (!ok) {
        err_ = ctl_.why();
    }
    return ok;
}

std::shared_ptr<cmd> leader::send_cmd(const ctrl & c, const int32_t * tokens) {
    std::shared_ptr<cmd> k = ctl_.send(c, tokens);
    if (k == nullptr) {
        err_ = ctl_.why();
    }
    return k;
}

bool leader::wait_cmd(cmd & k, double timeout_s) {
    const double w0 = ctl_.wait_s;
    const bool   ok = ctl_.wait(k, timeout_s);
    st_.back_wait_s += ctl_.wait_s - w0;
    if (!ok) {
        err_ = ctl_.why();
    }
    return ok;
}

// MUP1 piece 2: the front first (its refusal sends nothing), then the command without waiting for its X-ACK: the slot's
// next command collects it (read_begin's arrives in the same follower turn, so a request pays one round trip, not
// two); commands and groups go in order on the one channel, so the back samples with the new settings either way. A
// refusal there fails the slot's next call
run_status leader::set_sampling(int32_t slot, const request_options & o) {
    const run_status prior = settle(slot, true);
    if (prior != run_status::ok) {
        return prior;
    }
    const run_status rs = e_->set_sampling(slot, o);
    if (rs != run_status::ok) {
        err_ = e_->error();
        return rs;
    }
    ctrl c;
    c.op         = int32_t(op::set_sampling);
    c.slot       = slot;
    c.seed       = o.sampling.seed;
    c.temp       = o.sampling.temp;
    c.top_k      = o.sampling.top_k;
    c.top_p      = o.sampling.top_p;
    c.min_p      = o.sampling.min_p;
    c.max_tokens = o.max_tokens;
    c.n_ctx      = o.n_ctx;
    c.ignore_eos = o.ignore_eos ? 1 : 0;
    std::shared_ptr<cmd> k = send_cmd(c, nullptr);
    if (k == nullptr) {
        return run_status::error;
    }
    next_[size_t(slot)].valid = false;
    samp_[size_t(slot)]       = std::move(k);
    return run_status::ok;
}

// the slot's pending set_sampling X-ACK: waited for (wait) or taken when it has come; its refusal returned
run_status leader::settle(int32_t slot, bool wait) {
    if (slot < 0 || slot >= int32_t(samp_.size()) || samp_[size_t(slot)] == nullptr) {
        return run_status::ok;
    }
    cmd & k = *samp_[size_t(slot)];
    if (!k.answered) {
        bool done = false;
        if (wait ? !wait_cmd(k, ctrl_timeout_s) : !ctl_.test(k, done)) {
            ctl_.drop(k);
            samp_[size_t(slot)] = nullptr;
            err_                = ctl_.why();
            return run_status::error;
        }
        if (!wait && !done) {
            return run_status::ok;
        }
    }
    const ack a         = k.a;
    samp_[size_t(slot)] = nullptr;
    if (a.status != int32_t(run_status::ok)) {
        err_ = "set_sampling: the follower: " + why_of(a.why, sizeof a.why);
        return static_cast<run_status>(a.status);
    }
    return run_status::ok;
}

run_status leader::settle_all() {
    run_status out = run_status::ok;
    for (int32_t s = 0; s < int32_t(samp_.size()); ++s) {
        const run_status rs = settle(s, true);
        out                 = out == run_status::ok ? rs : out;
    }
    return out;
}

run_status leader::prefill(int32_t                           slot,
                           const int32_t *                   tokens,
                           int32_t                           n_tokens,
                           int32_t                           n_reuse,
                           int32_t &                         n_processed,
                           const std::vector<prompt_image> * images) {
    n_processed = 0;
    const run_status prior = settle(slot, true);  // MUP1 piece 2
    if (prior != run_status::ok) {
        return prior;
    }
    ctrl c;
    c.op                           = int32_t(op::prefill);
    c.slot                         = slot;
    c.a                            = n_reuse;
    c.graphs                       = model::prompt_graphs_on() ? 1 : 0;            // APF2
    const std::vector<int32_t> pay = read_payload(tokens, n_tokens, images, c.b);  // VIS1: the images' spans
    c.n_tokens                     = static_cast<int32_t>(pay.size());
    c.base                         = base_;
    // a read's chunks never reach 256 (plan_chunks: at most 8,192 rows a chunk). PRD1 item 9: 257, odd, so
    // consecutive reads start on alternate crossing buffers (a read's first chunk never waits on the last one's sends)
    base_ += 257;
    const std::shared_ptr<cmd> k = send_cmd(c, pay.data());
    if (k == nullptr) {
        return run_status::error;
    }
    const ack & a = k->a;
    x_->expect(slot, c.base);
    next_[size_t(slot)].valid = false;
    // the follower's answer may come first (its read failed): the front stops sending at its next chunk
    bool answered             = false;
    x_->abort                 = [&]() {
        if (!answered && !ctl_.test(*k, answered)) {
            answered = true;
        }
        return answered && k->answered && a.status != int32_t(run_status::ok);
    };
    const run_status rs = e_->prefill(slot, tokens, n_tokens, n_reuse, n_processed, {}, images);
    x_->abort           = nullptr;
    if (!k->answered && !wait_cmd(*k, 3600.0)) {
        ctl_.drop(*k);
        return run_status::error;
    }
    const bool ok = rs == run_status::ok && a.status == int32_t(run_status::ok) && a.a == n_processed;
    if (!ok) {
        // a failed read on either box: the crossings the follower did not take dropped there, and the leader's slot
        // starts again from token 0 (the boxes' states no longer agree)
        drain_read(slot, c.base);
        e_->truncate(slot, 0);
    } else {
        std::string e;
        if (!x_->drain(e)) {
            ctl_.lose(e);
            err_ = ctl_.why();
            return run_status::error;
        }
    }
    if (rs != run_status::ok) {
        err_ = e_->error();
        return rs;
    }
    if (a.status != int32_t(run_status::ok) || a.a != n_processed) {
        err_ = "prefill: the follower " +
               (a.status != int32_t(run_status::ok) ?
                    "failed: " + why_of(a.why, sizeof a.why) :
                    "processed " + std::to_string(a.a) + " tokens, the leader " + std::to_string(n_processed));
        return a.status != int32_t(run_status::ok) ? static_cast<run_status>(a.status) : run_status::error;
    }
    return run_status::ok;
}

int32_t leader::read_begin(int32_t                           slot,
                           const int32_t *                   tokens,
                           int32_t                           n_tokens,
                           int32_t                           n_reuse,
                           int32_t &                         n_processed,
                           const std::vector<prompt_image> * images) {
    n_processed = 0;
    // admission: the follower first (either box's pool or context refuses)
    ctrl c;
    c.op                           = int32_t(op::read_begin);
    c.slot                         = slot;
    c.a                            = n_reuse;
    c.graphs                       = model::prompt_graphs_on() ? 1 : 0;            // APF2
    const std::vector<int32_t> pay = read_payload(tokens, n_tokens, images, c.b);  // VIS1: the images' spans
    c.n_tokens                     = static_cast<int32_t>(pay.size());
    c.base                         = base_;
    base_ += 257;  // PRD1 item 9: odd, so consecutive reads alternate crossing buffers
    // MUP1 piece 2: the command first, then the slot's pending set_sampling X-ACK: both come in one follower turn
    const std::shared_ptr<cmd> k = send_cmd(c, pay.data());
    if (k == nullptr) {
        return -1;
    }
    const run_status prior = settle(slot, true);
    if (!k->answered && !wait_cmd(*k, ctrl_timeout_s)) {
        ctl_.drop(*k);
        return -1;
    }
    const ack a = k->a;
    if (prior != run_status::ok || a.status != 0) {
        const std::string why =
            prior != run_status::ok ? err_ : "read_begin: the follower: " + why_of(a.why, sizeof a.why);
        if (a.status == 0) {
            // the follower admitted the read: end its prepared read (cancelled before its first chunk)
            ctrl cc;
            cc.op   = int32_t(op::read_cancel);
            cc.slot = slot;
            ack ca;
            command(cc, nullptr, ca, ctrl_timeout_s);
        }
        err_ = why;
        return a.status != 0 ? a.status : -1;
    }
    x_->expect(slot, c.base);
    read_base_[size_t(slot)]  = c.base;
    next_[size_t(slot)].valid = false;
    const int32_t r           = e_->read_begin(slot, tokens, n_tokens, n_reuse, n_processed, images);
    if (r != 0 || a.a != n_processed) {
        err_ = r != 0 ? e_->error() :
                        "read_begin: the follower processes " + std::to_string(a.a) + " tokens, the leader " +
                            std::to_string(n_processed);
        // the follower's prepared read ends: cancelled before its first chunk
        ctrl cc;
        cc.op   = int32_t(op::read_cancel);
        cc.slot = slot;
        ack ca;
        command(cc, nullptr, ca, ctrl_timeout_s);
        if (r == 0) {
            e_->read_cancel(slot);
            e_->read_step(slot, 1, read_mode::shared);
        }
        return r != 0 ? r : -1;
    }
    read_acks_[size_t(slot)].clear();
    back_acct_[size_t(slot)] = read_info();
    return 0;
}

void leader::back_info(int32_t slot, read_info & out) const {
    out = slot >= 0 && size_t(slot) < back_acct_.size() ? back_acct_[size_t(slot)] : read_info();
}

read_status leader::read_step(int32_t slot, int32_t n_groups, read_mode mode, const read_budget * budget) {
    if (ctl_.lost()) {
        err_ = ctl_.why();
        return read_status::error;
    }
    // TTTS1: beside writers or other readers the back's chunks interleave with the step groups there, so the front's
    // end waits for their answers only in the lone mode (else yielded until they came; the drive's tests block)
    const bool block = mode == read_mode::lone || block_reads_;
    if (read_end_[size_t(slot)] >= 0) {
        const read_status f = read_finish(slot, block);
        return f == read_status::yielded ? read_status::front_done : f;
    }
    // MUP1 piece 3, exactly (PRD1 item 7): a small read beside writers: the front runs its chunk in one group
    // (engine::read_step) and the follower its back half whole (a = 0), at once when no earlier read waits there,
    // with or without a budget; the leader does not wait for it. MUP2: a larger read with a budget is budgeted on the
    // back too (a = 2), APF1: by the back's own writer-pause deadline under the P carried here
    const bool budgeted  = mode == read_mode::writers && budget != nullptr;
    const bool        whole     = mode == read_mode::lone || (mode == read_mode::writers && e_->small_read(slot));
    const read_status st = e_->read_step(slot, n_groups, mode, budget);
    err_                    = e_->error();
    auto & pend          = read_acks_[size_t(slot)];
    if (st == read_status::chunk_done || st == read_status::done || st == read_status::cancelled) {
        // the front's chunk is done and its crossing sent (or the read stopped): the follower runs the back's half,
        // whole (the lone mode) or interleaved
        ctrl c;
        c.op       = int32_t(op::read_step);
        c.slot     = slot;
        c.a                    = whole ? 0 : budgeted ? 2 : 1;
        c.pause_ms             = budgeted ? float(budget->pause_ms) : 0.0f;
        c.b        = st == read_status::cancelled ? 1 : 0;
        c.c        = int32_t(x_->sent(slot));  // a cancel: the chunks the front finished run on the back first
        std::shared_ptr<cmd> k = send_cmd(c, nullptr);
        if (k == nullptr) {
            return read_status::error;
        }
        pend.push_back(std::move(k));
    }
    if (st == read_status::yielded || st == read_status::chunk_done) {
        return st;
    }
    read_end_[size_t(slot)] = int32_t(st);
    // PRD1 item 9: the front is free at its end: front_done while the back's halves are out (read_poll then)
    const read_status f     = read_finish(slot, block || st == read_status::error);
    return f == read_status::yielded ? read_status::front_done : f;
}

read_status leader::read_poll(int32_t slot) {
    if (slot < 0 || size_t(slot) >= read_end_.size() || read_end_[size_t(slot)] < 0) {
        err_ = "read_poll: slot " + std::to_string(slot) + "'s front has not ended";
        return read_status::error;
    }
    const read_status f = read_finish(slot, ctl_.lost());
    return f == read_status::yielded ? read_status::front_done : f;
}

bool leader::read_ready(int32_t slot) {
    if (slot < 0 || size_t(slot) >= read_end_.size() || read_end_[size_t(slot)] < 0 || ctl_.lost()) {
        return true;
    }
    for (auto & k : read_acks_[size_t(slot)]) {
        bool done = false;
        if (!ctl_.test(*k, done)) {
            return true;  // lost: read_poll ends it
        }
        if (!done) {
            return false;
        }
    }
    return true;
}

read_status leader::read_finish(int32_t slot, bool block) {
    auto & pend = read_acks_[size_t(slot)];
    if (!block && !ctl_.lost()) {
        for (auto & k : pend) {
            bool done = false;
            if (!ctl_.test(*k, done)) {
                break;
            }
            if (!done) {
                return read_status::yielded;
            }
        }
    }
    const read_status st    = static_cast<read_status>(read_end_[size_t(slot)]);
    read_end_[size_t(slot)] = -1;
    // the read's end: every back half answered
    read_status out         = ctl_.lost() ? read_status::error : st;
    std::string e;
    // PRD1 item 9: only the crossing buffers this slot filled last (a later read's sends are its own)
    if (!ctl_.lost() && !x_->drain(slot, e)) {
        ctl_.lose(e);
        out = read_status::error;
    }
    for (size_t i = 0; i < pend.size(); ++i) {
        auto &      k = pend[i];
        const ack & a = k->a;
        if (!k->answered && (ctl_.lost() || !wait_cmd(*k, 3600.0))) {
            ctl_.drop(*k);  // its answer no longer wanted: retired
            out = read_status::error;
            continue;
        }
        // MUP2 piece 4: the back's account so far (answers in chunk order: the last is the read's)
        if (a.acct_i[0] > 0) {
            read_info & b   = back_acct_[size_t(slot)];
            b.unit_ms       = a.acct[0];
            b.floor_ms      = a.acct[1];
            b.budget_n      = a.acct[2] >= 0 ? 1 : 0;
            b.budget_ms_sum = a.acct[2];
            b.pause_ms       = int32_t(a.acct[3]);
            b.service_s     = a.acct[4] / 1e3;
            b.units_run     = a.acct_i[0];
            b.unit_steps    = a.acct_i[1];
            b.most_units    = a.acct_i[2];
            b.floor_units   = a.acct_i[3];
            b.piece_min     = a.acct_i[4];
            b.piece_max     = a.acct_i[5];
            b.budget_misses = a.acct_i[6];
            b.below_floor   = a.acct_i[7] != 0;
            // APF1: the back's pacing over the read
            b.budget_zero    = a.acct_i[8];
            b.budget_unknown = a.acct_i[9];
            pace_from_ack(a, b.pace_ns);
        }
        // the back's answers in chunk order: chunk_done for every chunk but the last, whose answer is the read's end
        // (PRD1 item 14: a cancel after finished chunks answers chunk_done for those, then cancelled)
        const read_status want = i + 1 < pend.size() ? read_status::chunk_done : out;
        if (a.status != int32_t(want) && (out == read_status::done || out == read_status::cancelled)) {
            err_ = "read: the follower's back half: " + why_of(a.why, sizeof a.why);
            out  = read_status::error;
        }
    }
    pend.clear();
    if (out != read_status::done) {
        drain_read(slot, read_base_[size_t(slot)]);
        if (out == read_status::error) {
            e_->truncate(slot, 0);
        }
    }
    return out;
}

void leader::read_cancel(int32_t slot) {
    settle(slot, false);    // MUP1 piece 2: taken if it came (its refusal is the request's end anyway)
    e_->read_cancel(slot);  // the front stops at its chunk's end; read_step tells the follower
}

run_status leader::truncate(int32_t slot, int32_t n_keep) {
    const run_status prior = settle(slot, true);  // MUP1 piece 2
    if (prior != run_status::ok) {
        return prior;
    }
    // the leader's cut first; the follower's with it (c.b: the leader's slot in place); a slot either box could not
    // keep in place restores at the next prefill on both, so their reads start at the same checkpoint
    next_[size_t(slot)].valid = false;
    // TTTS8: a slot's checkpoints change on both boxes together: a box whose cut failed dropped its own (its state
    // unknown, engine::truncate), so the other drops them too (c.c: the leader did; the ack's b: the follower did) and
    // both read the next prompt from the same point; read_begin's count check stays the guard
    prefiller &       pf      = e_->prefill_driver();
    const int32_t     had     = pf.book().count_owner(slot);
    const run_status  rs      = e_->truncate(slot, n_keep);
    const std::string lerr    = e_->error();
    const bool        dropped = had > 0 && pf.book().count_owner(slot) == 0;
    ctrl              c;
    c.op   = int32_t(op::truncate);
    c.slot = slot;
    c.a    = n_keep;
    c.b    = rs == run_status::ok && e_->in_place(slot) ? 1 : 0;
    c.c    = dropped ? 1 : 0;
    ack a;
    if (!command(c, nullptr, a, ctrl_timeout_s)) {
        return run_status::error;
    }
    if (a.status != int32_t(run_status::ok)) {
        err_ = "truncate: the follower: " + why_of(a.why, sizeof a.why);
        return static_cast<run_status>(a.status);
    }
    if (a.b != 0 && !dropped) {
        const int32_t n = pf.book().count_owner(slot);
        pf.drop_owner(slot);
        fprintf(
            stderr,
            "FNFL split: slot %d: truncate to %d: the follower dropped the slot's checkpoints (its cut failed); the "
            "leader drops its %d, so both read the next prompt from token 0\n",
            slot, n_keep, n);
        fflush(stderr);
    }
    if (a.a == 0) {
        e_->to_restore(slot);
    }
    err_ = lerr;
    return rs;
}

run_status leader::release_slot(int32_t slot) {
    settle(slot, true);  // MUP1 piece 2: the slot ends; a refused setting does not matter now
    ctrl c;
    c.op   = int32_t(op::release_slot);
    c.slot = slot;
    ack a;
    if (!command(c, nullptr, a, ctrl_timeout_s)) {
        return run_status::error;
    }
    next_[size_t(slot)].valid = false;
    const run_status rs       = e_->release_slot(slot);
    err_                      = e_->error();
    return a.status != int32_t(run_status::ok) ? static_cast<run_status>(a.status) : rs;
}

run_status leader::set_draft(int32_t d, bool sparse) {
    const run_status prior = settle_all();  // MUP1 piece 2
    if (prior != run_status::ok) {
        return prior;
    }
    ctrl c;
    c.op = int32_t(op::set_draft);
    c.a  = d;
    c.b  = sparse ? 1 : 0;
    ack a;
    if (!command(c, nullptr, a, ctrl_timeout_s)) {
        return run_status::error;
    }
    for (next_step & nx : next_) {
        nx.valid = false;
    }
    const run_status rs = e_->set_draft(d, sparse);
    if (a.status != int32_t(run_status::ok)) {
        err_ = "set_draft: the follower: " + why_of(a.why, sizeof a.why);
        return static_cast<run_status>(a.status);
    }
    err_ = e_->error();
    return rs;
}

// the front first (its refusals: the slot, the range, a live read), then the back; a refusal there restores the front's
// old offset, so the boxes never disagree
run_status leader::set_pos_offset(int32_t slot, int32_t off) {
    const run_status prior = settle(slot, true);  // MUP1 piece 2
    if (prior != run_status::ok) {
        return prior;
    }
    const int32_t    old = e_->pos_offset(slot);
    const run_status rs  = e_->set_pos_offset(slot, off);
    if (rs != run_status::ok) {
        err_ = e_->error();
        return rs;
    }
    if (slot < int32_t(next_.size())) {
        next_[size_t(slot)].valid = false;  // a next step drafted at the old positions is not used
    }
    ctrl c;
    c.op   = int32_t(op::pos_offset);
    c.slot = slot;
    c.a    = off;
    ack a;
    if (!command(c, nullptr, a, ctrl_timeout_s) || a.status != int32_t(run_status::ok)) {
        e_->set_pos_offset(slot, old);
        err_ = "set_pos_offset: the follower: " + why_of(a.why, sizeof a.why);
        return a.status != int32_t(run_status::ok) ? static_cast<run_status>(a.status) : run_status::error;
    }
    return run_status::ok;
}

bool leader::follower_pool(pool_info & out) {
    settle_all();  // MUP1 piece 2
    ctrl c;
    c.op = int32_t(op::pool);
    ack a;
    if (!command(c, nullptr, a, ctrl_timeout_s) || a.status != 0) {
        return false;
    }
    out                = pool_info();
    out.on             = true;
    out.pages          = a.a;
    out.free_pages     = a.b;
    out.page_cells     = a.c;
    out.capacity_cells = a.x;
    return true;
}

void joint_refusals::refuse(const char * tag, const std::string & cause, bool capture_failed, const char * tail) {
    off                   = off || capture_failed;
    const std::string key = std::string(tag) + cause + tail;
    if (said.insert(key).second) {
        fprintf(stderr, "%s: %s; %s%s\n", tag, cause.c_str(), tail,
                capture_failed ? " (joint decode off from now on)" : " (said once a cause)");
        fflush(stderr);
    }
}

void leader::bye() {
    settle_all();  // MUP1 piece 2: every X-ACK taken (or dropped on a lost stage) before the link closes
    if (ctl_.lost()) {
        p_->close();
        return;
    }
    ctrl c;
    c.op = int32_t(op::bye);
    ack a;
    command(c, nullptr, a, 10.0);
}

std::unique_ptr<leader::pending> leader::group_begin(const int32_t * slots, int32_t n, int32_t group) {
    nvtx_range nr("leader.group_begin");
    auto       g             = std::make_unique<pending>();
    g->group                 = group;
    g->n                     = n;
    g->t0                    = clk::now();
    const engine_config & cf = e_->config();
    // each member's checks and cells, and its next step as the last X-STEP left it
    std::vector<int32_t>  lanes;
    nvtx_phase            ph("leader.enter");
    for (int32_t k = 0; k < n; ++k) {
        member & m = g->m[k];
        m.slot     = slots[k];
        g->st[k]   = ctl_.lost() ? run_status::error : settle(slots[k], false);  // MUP1 piece 2: a refused setting
        if (g->st[k] != run_status::ok) {
            g->why[k] = ctl_.lost() ? ctl_.why() : err_;
            m.kind    = -1;  // not sent
            continue;
        }
        g->st[k] = e_->split_enter(slots[k], g->b[k]);
        if (g->st[k] != run_status::ok) {
            g->why[k] = ctl_.lost() ? ctl_.why() : e_->error();
            m.kind    = -1;  // not sent
            continue;
        }
        const next_step & nx = next_[size_t(slots[k])];
        if (!nx.valid) {
            m.kind = int32_t(mkind::begin);
        } else if (nx.kind == int32_t(mkind::generate)) {
            m.kind = nx.kind;
            m.rows = 1;
        } else {
            m.kind = nx.kind;
            m.d    = nx.d;
            m.rows = nx.d + 1;
            std::copy(nx.prop, nx.prop + 4, m.prop);
            // FMU2's lane rule: a speculative step at the full d (FMS13: any d, the round's own)
            if (nx.d == cf.spec_d) {
                lanes.push_back(k);
            }
        }
    }
    // the slots' boundary buffers are refilled by the front rounds below: the slot's previous group ended (its sends
    // finished or retired) before the pipeline let it begin this one
    for (int32_t k = 0; k < n; ++k) {
        if (g->m[k].kind > int32_t(mkind::begin)) {
            for (int32_t card = 0; card < 2; ++card) {
                life::reuse(cf.stage_buf[size_t(g->m[k].slot)][size_t(card)], size_t(model::stage_bytes(g->m[k].rows)),
                            "boundary buffer");
            }
        }
    }
    ph.next("leader.front");
    // the front rounds: the lanes' joint round, then every other member's round alone
    joint_frame * jf    = e_->split_joint();
    bool          joint = false;
    if (lanes.size() >= 2 && jf != nullptr && !joint_verify_.off && lanes.size() <= size_t(fn::xchg::max_lanes)) {
        target *            tt[fn::xchg::max_lanes] = {};
        int32_t             d[fn::xchg::max_lanes] = {}, bud[fn::xchg::max_lanes] = {};
        const int32_t *     prop[fn::xchg::max_lanes] = {};
        target::front_round fr[fn::xchg::max_lanes];
        round_result        rf[fn::xchg::max_lanes];
        bool                ok[fn::xchg::max_lanes] = {};
        std::string         why[fn::xchg::max_lanes], jerr;
        for (size_t i = 0; i < lanes.size(); ++i) {
            const int32_t k = lanes[i];
            tt[i]           = &e_->test_target(g->m[k].slot);
            d[i]            = g->m[k].d;
            bud[i]          = g->b[k];
            prop[i]         = g->m[k].prop;
        }
        bool staged = false;
        joint       = jf->front_verify(tt, int32_t(lanes.size()), d, prop, bud, fr, rf, ok, why, staged, jerr);
        if (!joint && staged) {
            for (int32_t k : lanes) {
                g->st[k]     = run_status::error;
                g->why[k]    = "joint front round: " + jerr;
                g->m[k].kind = -1;
            }
        } else if (joint) {
            for (size_t i = 0; i < lanes.size(); ++i) {
                const int32_t k = lanes[i];
                g->fr[k]        = fr[i];
                g->fronted[k]   = ok[i];
                g->m[k].joint   = 1;
                if (!ok[i]) {
                    g->st[k]     = run_status::error;
                    g->why[k]    = why[i];
                    g->m[k].kind = -1;
                }
            }
        } else {
            joint_verify_.refuse("FNFL split", jerr, jf->capture_failed(), "these slots verify alone");
        }
    }
    for (int32_t k = 0; k < n; ++k) {
        member & m = g->m[k];
        if (m.kind <= int32_t(mkind::begin) || g->fronted[k] || m.joint != 0) {
            continue;
        }
        target &      t = e_->test_target(m.slot);
        round_result  r;
        std::string   e;
        const int32_t kind =
            m.kind == int32_t(mkind::generate) ? int32_t(fn::round::kind::generate) : int32_t(fn::round::kind::verify);
        if (!t.front_begin(kind, m.rows, m.prop, g->b[k], g->fr[k], e) || !t.front_wait(g->fr[k], r, e)) {
            g->st[k]  = run_status::error;
            g->why[k] = "front round: " + e;
            m.kind    = -1;
            continue;
        }
        g->fronted[k] = true;
    }
    ph.next("leader.post");
    dispatch(*g, group);
    return g;
}

std::unique_ptr<leader::pending> leader::group_begin_forced(const int32_t * slots,
                                                            int32_t         n,
                                                            int32_t         rows,
                                                            int32_t         group) {
    auto g   = std::make_unique<pending>();
    g->group = group;
    g->n     = n;
    g->t0    = clk::now();
    std::vector<target *> tt;
    for (int32_t k = 0; k < n; ++k) {
        member & m = g->m[k];
        m.slot     = slots[k];
        m.kind     = int32_t(mkind::forced);
        m.rows     = rows;
        g->st[k]   = ctl_.lost() ? run_status::error : run_status::ok;
        g->why[k]  = ctl_.lost() ? ctl_.why() : std::string();
        tt.push_back(&e_->test_target(slots[k]));
    }
    if (ctl_.lost()) {
        for (int32_t k = 0; k < n; ++k) {
            g->m[k].kind = -1;
        }
        return g;
    }
    for (int32_t k = 0; k < n; ++k) {  // refilled by the front rounds: the slot's previous group ended
        for (int32_t card = 0; card < 2; ++card) {
            life::reuse(e_->config().stage_buf[size_t(slots[k])][size_t(card)], size_t(model::stage_bytes(rows)),
                        "boundary buffer");
        }
    }
    joint_frame * jf = e_->split_joint();
    if (n >= 2 && jf != nullptr) {
        round_result rf[fn::xchg::max_lanes];
        bool         ok[fn::xchg::max_lanes] = {};
        std::string  why[fn::xchg::max_lanes], jerr;
        bool         staged = false;
        const bool   joint  = jf->front_forced(tt.data(), n, rows, g->fr, rf, ok, why, staged, jerr);
        for (int32_t k = 0; k < n; ++k) {
            g->fronted[k] = joint && ok[k];
            g->m[k].joint = joint ? 1 : 0;
            if (!g->fronted[k]) {
                g->st[k]     = run_status::error;
                g->why[k]    = "joint front forced: " + (joint ? why[k] : jerr);
                g->m[k].kind = -1;
            }
        }
    } else {
        for (int32_t k = 0; k < n; ++k) {
            round_result r;
            std::string  e;
            if (!tt[size_t(k)]->front_begin(static_cast<int32_t>(fn::round::kind::forced), rows, nullptr, rows,
                                            g->fr[k], e) ||
                !tt[size_t(k)]->front_wait(g->fr[k], r, e)) {
                g->st[k]     = run_status::error;
                g->why[k]    = "front forced: " + e;
                g->m[k].kind = -1;
                continue;
            }
            g->fronted[k] = true;
        }
    }
    dispatch(*g, group);
    return g;
}

void leader::forced_groups(const int32_t *  slots,
                           const int32_t *  sizes,
                           int32_t          n_groups,
                           int32_t          rows,
                           joint_step_out * res) {
    settle_all();  // MUP1 piece 2 (the drive's tests)
    std::vector<std::unique_ptr<pending>> g;
    int32_t                               at = 0;
    for (int32_t q = 0; q < n_groups; ++q) {
        g.push_back(group_begin_forced(slots + at, sizes[q], rows, q));
        at += sizes[q];
    }
    at = 0;
    for (int32_t q = 0; q < n_groups; ++q) {
        group_end(*g[size_t(q)], res + at);
        at += sizes[q];
    }
}

bool leader::probe(int32_t slot, int32_t what, std::vector<uint8_t> & out) {
    out.clear();
    settle(slot, true);  // MUP1 piece 2
    ctrl c;
    c.op   = int32_t(op::probe);
    c.slot = slot;
    c.a    = what;
    ack a;
    if (!command(c, nullptr, a, 3600.0)) {
        return false;
    }
    if (a.status != int32_t(run_status::ok) || a.a < 0) {
        err_ = "probe: the follower: " + why_of(a.why, sizeof a.why);
        return false;
    }
    const uint32_t seq = ctl_.last_seq();  // the command's
    if (a.a == 0) {
        return true;
    }
    // the answer's bytes land in a heap record its receive keeps, then are copied out
    auto          got = std::make_shared<std::vector<uint8_t>>(size_t(a.a), 0);
    std::string   e;
    const int32_t op = p_->recv(probe_tag(seq), got->data(), got->size(), got, e);
    if (op < 0 || !p_->wait(op, e, ctrl_timeout_s)) {
        ctl_.lose(e);
        err_ = ctl_.why();
        return false;
    }
    out = *got;
    return true;
}

run_status leader::reconfigure(ctrl c) {
    settle_all();  // MUP1 piece 2
    c.op   = int32_t(op::reconfigure);
    c.slot = -1;
    ack a;
    if (!command(c, nullptr, a, 3600.0)) {
        return run_status::error;
    }
    if (a.status != int32_t(run_status::ok)) {
        err_ = "reconfigure: the follower: " + why_of(a.why, sizeof a.why);
        return static_cast<run_status>(a.status);
    }
    return run_status::ok;
}

void leader::rebind(engine & e) {
    settle_all();  // MUP1 piece 2
    e_ = &e;
    next_.assign(size_t(e.n_slots()), next_step());
    samp_.assign(size_t(e.n_slots()), nullptr);
    read_acks_.assign(size_t(e.n_slots()), {});
    read_base_.assign(size_t(e.n_slots()), 0);
    read_end_.assign(size_t(e.n_slots()), -1);
    back_acct_.assign(size_t(e.n_slots()), read_info());
    logits_.assign(size_t(e.n_slots()), {});
}

// the routes' counts in test_route's order (stage_msg.h)
using route_field                                            = int64_t model::prompt_path::route_counts::*;
constexpr route_field       route_fields[route_count_fields] = {
    &model::prompt_path::route_counts::dense_ladder,  &model::prompt_path::route_counts::dense_retained,
    &model::prompt_path::route_counts::dense_split,   &model::prompt_path::route_counts::hc_ladder,
    &model::prompt_path::route_counts::hc_retained,   &model::prompt_path::route_counts::ex_grouped,
    &model::prompt_path::route_counts::ex_per_task,   &model::prompt_path::route_counts::exd_grouped,
    &model::prompt_path::route_counts::exd_per_task,  &model::prompt_path::route_counts::graph_units,
    &model::prompt_path::route_counts::eager_units,   &model::prompt_path::route_counts::graph_captures,
    &model::prompt_path::route_counts::graph_updated, &model::prompt_path::route_counts::qsa,
    &model::prompt_path::route_counts::gdn,           &model::prompt_path::route_counts::ple,
};

run_status leader::test_force_route(const model::prompt_route_force & f, model::prompt_path::route_counts * back) {
    const run_status prior = settle_all();
    if (prior != run_status::ok) {
        return prior;
    }
    e_->prefill_driver().test_force_route(f);
    ctrl c;
    c.op   = int32_t(op::test_route);
    c.a    = f.dense;
    c.b    = f.hc;
    c.c    = f.expert;
    c.base = f.graph;
    ack a;
    if (!command(c, nullptr, a, ctrl_timeout_s)) {
        return run_status::error;
    }
    if (a.status != int32_t(run_status::ok)) {
        err_ = "test_force_route: the follower: " + why_of(a.why, sizeof a.why);
        return static_cast<run_status>(a.status);
    }
    if (back != nullptr) {
        *back = model::prompt_path::route_counts{};
        for (int32_t i = 0; i < route_count_fields; ++i) {
            back->*route_fields[i] = a.acct_i[i];
        }
    }
    return run_status::ok;
}

run_status leader::test_prompt(int32_t slot, const std::vector<int32_t> & tokens, bool keep) {
    target &    t = e_->test_target(slot);
    std::string e;
    if (keep) {
        if (!t.keep_processed(e)) {
            err_ = "test_prompt: the front's kept tokens: " + e;
            return run_status::error;
        }
        t.append(tokens);
    } else {
        if (!t.reset(e)) {
            err_ = "test_prompt: the front's reset: " + e;
            return run_status::error;
        }
        t.set_prompt(tokens);
    }
    next_[size_t(slot)].valid = false;
    ctrl c;
    c.op       = int32_t(op::test_prompt);
    c.slot     = slot;
    c.a        = keep ? 1 : 0;
    c.n_tokens = int32_t(tokens.size());
    ack a;
    if (!command(c, tokens.data(), a, ctrl_timeout_s)) {
        return run_status::error;
    }
    if (a.status != int32_t(run_status::ok)) {
        err_ = "test_prompt: the follower: " + why_of(a.why, sizeof a.why);
        return static_cast<run_status>(a.status);
    }
    return run_status::ok;
}

// the group to the follower: its command, the members' boundaries, the answers' receives (fn/runtime/stage_wire.h)
void leader::dispatch(pending & g, int32_t group) {
    nvtx_range nr("leader.dispatch");
    int32_t nm = 0;
    for (int32_t k = 0; k < g.n; ++k) {
        nm += g.m[k].kind >= 0 ? 1 : 0;
    }
    if (nm == 0) {
        return;
    }
    if (ctl_.lost()) {
        for (int32_t k = 0; k < g.n; ++k) {
            g.st[k]  = run_status::error;
            g.why[k] = ctl_.why();
        }
        return;
    }
    const auto & buf = e_->config().stage_buf;
    boundary_src src[max_members];
    size_t       logits_n[max_members] = {};
    for (int32_t k = 0; k < g.n; ++k) {
        if (g.m[k].kind == int32_t(mkind::forced)) {
            logits_n[k] = size_t(2) * size_t(g.m[k].rows) * size_t(fn::head::n_vocab_card);
        }
        if (g.fronted[k]) {
            src[k].buf[0] = buf[size_t(g.m[k].slot)][0];
            src[k].buf[1] = buf[size_t(g.m[k].slot)][1];
            src[k].bytes  = size_t(model::stage_bytes(g.m[k].rows));
        }
    }
    // APF1: the pacing the follower needs: P, this step's front service, the front's groups and their generations
    ctrl & c   = g.w->c;
    c.pause_ms = float(pace_pause_);
    c.front_ms = float(since(g.t0) * 1e3);
    c.live     = pace_live_ | (group >= 0 && group < 2 ? 1u << uint32_t(group) : 0u);
    c.gen[0]   = pace_gen_[0];
    c.gen[1]   = pace_gen_[1];
    post_group(ctl_, g.w, group, g.m, g.n, src, logits_n, st_);
}

void leader::set_pace(int32_t pause_ms, uint32_t live, const int32_t * gen) {
    pace_pause_  = pause_ms;
    pace_live_   = live;
    pace_gen_[0] = gen[0];
    pace_gen_[1] = gen[1];
}

void leader::group_end(pending & g, joint_step_out * res) {
    nvtx_range   nr("leader.group_end");
    std::string  e;
    const auto   t0   = clk::now();
    group_wire & w    = *g.w;
    last_back_ms_     = -1;  // APF1: the group's back stage, from its members' answers
    const bool   sent = [&] {
        nvtx_range nw("leader.send_wait");
        return group_sends(ctl_, w, step_timeout_s);
    }();
    for (int32_t k = 0; k < g.n; ++k) {
        res[k]     = joint_step_out();
        member & m = g.m[k];
        if (m.kind < 0 || !sent) {
            if (g.fronted[k]) {
                e_->test_target(m.slot).front_abandon(g.fr[k]);
            }
            const std::string why = !sent && m.kind >= 0 ? ctl_.why() : g.why[k];
            step_info         s;
            res[k].st = g.st[k] == run_status::ok ? run_status::error : g.st[k];
            if (g.st[k] == run_status::ok && m.kind != int32_t(mkind::forced)) {
                e_->split_leave(m.slot, false, s, res[k].out, res[k].n_out, res[k].end);
            }
            res[k].why                  = why;
            next_[size_t(m.slot)].valid = false;
            continue;
        }
        bool got = false;
        {
            nvtx_range nw("leader.xstep_wait");
            got = group_answer(ctl_, w, k, step_timeout_s, e);
        }
        if (!got) {
            if (g.fronted[k]) {
                e_->test_target(m.slot).front_abandon(g.fr[k]);
            }
            step_info s;
            if (m.kind != int32_t(mkind::forced)) {
                e_->split_leave(m.slot, false, s, res[k].out, res[k].n_out, res[k].end);
            }
            res[k].st                   = run_status::error;
            res[k].why                  = ctl_.why();
            next_[size_t(m.slot)].valid = false;
            continue;
        }
        const step_out & o  = w.out[k];
        target &         t  = e_->test_target(m.slot);
        next_step &      nx = next_[size_t(m.slot)];
        last_back_ms_       = std::max(last_back_ms_, double(o.back_ms));
        nx.valid            = o.status == int32_t(run_status::ok) && o.next_kind != 0;
        nx.kind             = o.next_kind;
        nx.d                = o.next_d;
        std::copy(o.next_prop, o.next_prop + 4, nx.prop);
        if (o.status != int32_t(run_status::ok)) {
            if (g.fronted[k]) {
                t.front_abandon(g.fr[k]);
            }
            step_info s;
            if (m.kind != int32_t(mkind::forced)) {
                e_->split_leave(m.slot, false, s, res[k].out, res[k].n_out, res[k].end);
            }
            res[k].st  = static_cast<run_status>(o.status);
            res[k].why = "the back: " + why_of(o.why, sizeof o.why);
            nx.valid   = false;
            continue;
        }
        if (!o.ran) {
            res[k].st    = run_status::ok;  // a begin that only drafted: no step yet (the caller steps it again)
            res[k].n_out = 0;
            continue;
        }
        if (m.kind == int32_t(mkind::forced)) {
            // the drive's tests: the forced rows committed as the back's, its logits kept for the caller
            const int32_t kf  = fn::round::commit_versions(static_cast<int32_t>(fn::round::kind::forced), m.rows, 0, 0);
            const bool    fok = t.front_end(g.fr[k], kf, nullptr, 0, -1, 0, e);
            life::disown(w.lg[k].data(), "forced logits");  // its receive finished: the caller's now
            logits_[size_t(m.slot)] = std::move(w.lg[k]);
            res[k].joint            = o.joint != 0;
            res[k].st               = fok ? run_status::ok : run_status::error;
            res[k].why              = fok ? std::string() : "the front's forced commit: " + e;
            nx.valid                = false;
            continue;
        }
        bool ok = true;
        if (m.kind == int32_t(mkind::begin)) {
            ok = o.kind == 0 && t.front_bootstrap(o.out[0], e);  // the bootstrap: a whole step on the back
        } else {
            ok = t.front_end(g.fr[k], o.round_n_out, o.round_out, o.round_n_out, o.cut ? o.cut_rows : -1, o.n_keep, e);
        }
        step_info s = info_of(o);
        if (!ok) {
            fprintf(stderr, "FNFL split: slot %d: the front's commit of the back's verdict failed: %s\n", m.slot,
                    e.c_str());
            nx.valid = false;
        }
        res[k].joint = o.joint != 0;
        res[k].st    = e_->split_leave(m.slot, ok, s, res[k].out, res[k].n_out, res[k].end);
        res[k].why   = res[k].st == run_status::ok ? std::string() : e_->error();
    }
    group_close(ctl_, w);
    st_.back_wait_s += since(t0);
}

void leader::step_group(const int32_t * slots, int32_t n, joint_step_out * res) {
    if (n < 1 || n > max_members) {
        for (int32_t k = 0; k < n; ++k) {
            res[k]     = joint_step_out();
            res[k].st  = run_status::bad_request;
            res[k].why = "step_group: 1 .. 8 slots";
        }
        return;
    }
    // members whose next step is not known begin first (the bootstrap, or the next draft only), then every member
    // without a step yet runs its step
    std::vector<int32_t> first, idx;
    for (int32_t k = 0; k < n; ++k) {
        if (!next_[size_t(slots[k])].valid) {
            first.push_back(slots[k]);
            idx.push_back(k);
        }
    }
    bool done[max_members] = {};
    if (!first.empty()) {
        joint_step_out r[max_members];
        auto           g = group_begin(first.data(), int32_t(first.size()), 0);
        group_end(*g, r);
        for (size_t i = 0; i < first.size(); ++i) {
            const int32_t k = idx[i];
            res[k]          = r[i];
            done[k]         = r[i].st != run_status::ok || r[i].n_out > 0;
        }
    }
    std::vector<int32_t> rest, ridx;
    for (int32_t k = 0; k < n; ++k) {
        if (!done[k]) {
            rest.push_back(slots[k]);
            ridx.push_back(k);
        }
    }
    if (rest.empty()) {
        return;
    }
    joint_step_out r[max_members];
    auto           g = group_begin(rest.data(), int32_t(rest.size()), 0);
    group_end(*g, r);
    for (size_t i = 0; i < rest.size(); ++i) {
        res[ridx[i]] = r[i];
    }
}

// ---- the follower ----

// ---- the leader's pipeline ----

pipeline::pipeline(leader & l) : l_(&l) {}

pipeline::~pipeline() = default;

uint32_t pipeline::live_mask() const {
    uint32_t m = 0;
    for (int32_t g = 0; g < 2; ++g) {
        if (!core_.group(g).empty() || core_.in_flight(g)) {
            m |= 1u << uint32_t(g);
        }
    }
    return m;
}

void pipeline::end_group(int32_t g) {
    joint_step_out out[max_members];
    l_->group_end(*fl_[g], out);
    pace_.answered(g, pace::clock_ms(), l_->last_back_ms());  // APF1: the group's observed completion
    for (size_t k = 0; k < members_[g].size(); ++k) {
        // a begin that only drafted is no step (its slot steps again at its next turn); every other result queues
        // behind the slot's earlier ones
        if (out[k].st == run_status::ok && out[k].n_out == 0 && !out[k].end) {
            continue;
        }
        kept_.push_back({ members_[g][k], out[k] });
    }
    fl_[g].reset();
    members_[g].clear();
    pipe::action x;
    x.a = pipe::act::end;
    x.g = g;
    core_.done(x);
}

void pipeline::step(const int32_t * slots, int32_t n, joint_step_out * res, bool * done) {
    nvtx_range           nr("pipe.step");
    bool                 any_kept = false;
    std::vector<int32_t> w;
    for (int32_t k = 0; k < n; ++k) {
        if (queued(slots[k]) > 0) {
            any_kept = true;
        } else if (!ended(slots[k])) {
            w.push_back(slots[k]);
        }
    }
    core_.set_writers(w);
    while (!any_kept) {
        const pipe::action x = core_.next();
        if (x.a == pipe::act::begin) {
            // APF1: the group's front stage (its admission, a new membership's at this safe boundary) and what its
            // command carries to the follower
            pace_.front_begin(x.g, x.members.data(), int32_t(x.members.size()), pace::clock_ms());
            const int32_t gen[2] = { int32_t(pace_.group_gen(0)), int32_t(pace_.group_gen(1)) };
            l_->set_pace(pace_.pause(), live_mask() | (1u << uint32_t(x.g)), gen);
            fl_[x.g]      = l_->group_begin(x.members.data(), int32_t(x.members.size()), x.g);
            pace_.front_sent(x.g, pace::clock_ms());
            members_[x.g] = x.members;
            core_.done(x);
            continue;
        }
        if (x.a == pipe::act::end) {
            end_group(x.g);
        }
        break;  // an end, or nothing to do (front reads are the shell's own calls)
    }
    // APF1: a group the core emptied (its writers left, none in flight) has no timing left
    for (int32_t g = 0; g < 2; ++g) {
        if (core_.group(g).empty() && !core_.in_flight(g)) {
            pace_.clear(g);
        }
    }
    for (int32_t k = 0; k < n; ++k) {
        done[k] = take(slots[k], res[k]);
        if (!done[k] && ended(slots[k])) {
            res[k]     = joint_step_out();
            res[k].st  = run_status::bad_request;
            res[k].why = "slot " + std::to_string(slots[k]) + ": a step after its request ended";
            done[k]    = true;
        }
    }
}

bool pipeline::take(int32_t slot, joint_step_out & res) {
    for (auto it = kept_.begin(); it != kept_.end(); ++it) {
        if (it->first == slot) {
            res = it->second;
            kept_.erase(it);
            delivered(slot, res);
            return true;
        }
    }
    return false;
}

void pipeline::delivered(int32_t slot, const joint_step_out & res) {
    if (slot >= 0 && slot < pipe::max_writers && (res.end || res.st != run_status::ok)) {
        ended_[slot] = true;
    }
}

bool pipeline::ended(int32_t slot) const {
    return slot >= 0 && slot < pipe::max_writers && ended_[slot];
}

int32_t pipeline::queued(int32_t slot) const {
    return int32_t(std::count_if(kept_.begin(), kept_.end(), [slot](const auto & k) { return k.first == slot; }));
}

void pipeline::settle(int32_t slot) {
    for (int32_t g = 0; g < 2; ++g) {
        if (fl_[g] != nullptr && std::find(members_[g].begin(), members_[g].end(), slot) != members_[g].end()) {
            end_group(g);
        }
    }
}

void pipeline::settle_all() {
    for (int32_t g = 0; g < 2; ++g) {
        if (fl_[g] != nullptr) {
            end_group(g);
        }
    }
}

int32_t pipeline::end_request(int32_t slot, const char * call) {
    settle(slot);
    pace_.slot_ended(slot);  // APF1: a new member when it next joins
    const int32_t n = queued(slot);
    kept_.erase(std::remove_if(kept_.begin(), kept_.end(), [slot](const auto & k) { return k.first == slot; }),
                kept_.end());
    if (slot >= 0 && slot < pipe::max_writers) {
        ended_[slot] = false;
    }
    if (n > 0) {
        fprintf(stderr, "FNFL split: slot %d: %d finished step%s not delivered dropped at %s (its request ended)\n",
                slot, n, n == 1 ? "" : "s", call);
        fflush(stderr);
    }
    return n;
}

void pipeline::step_one(int32_t slot, joint_step_out & res) {
    nvtx_range nr("pipe.step_one");
    res = joint_step_out();
    settle_all();
    // APF1: a lone writer steps outside the pipeline (no reader beside it): the groups' timing is cleared, and the
    // follower clears its own from this step's command
    pace_.clear_all();
    const int32_t gen[2] = { int32_t(pace_.group_gen(0)), int32_t(pace_.group_gen(1)) };
    l_->set_pace(pace_.pause(), 0, gen);
    if (take(slot, res)) {
        return;
    }
    if (ended(slot)) {
        res.st  = run_status::bad_request;
        res.why = "slot " + std::to_string(slot) + ": a step after its request ended";
        return;
    }
    l_->step_group(&slot, 1, &res);
    delivered(slot, res);
}

run_status pipeline::set_pos_offset(int32_t slot, int32_t off) {
    end_request(slot, "set_pos_offset");
    return l_->set_pos_offset(slot, off);
}

run_status pipeline::set_sampling(int32_t slot, const request_options & o) {
    nvtx_range nr("pipe.set_sampling");
    end_request(slot, "set_sampling");
    return l_->set_sampling(slot, o);
}

run_status pipeline::prefill(int32_t                           slot,
                             const int32_t *                   tokens,
                             int32_t                           n_tokens,
                             int32_t                           n_reuse,
                             int32_t &                         n_processed,
                             const std::vector<prompt_image> * images) {
    nvtx_range nr("pipe.prefill");
    end_request(slot, "prefill");
    return l_->prefill(slot, tokens, n_tokens, n_reuse, n_processed, images);
}

int32_t pipeline::read_begin(int32_t                           slot,
                             const int32_t *                   tokens,
                             int32_t                           n_tokens,
                             int32_t                           n_reuse,
                             int32_t &                         n_processed,
                             const std::vector<prompt_image> * images) {
    end_request(slot, "read_begin");
    return l_->read_begin(slot, tokens, n_tokens, n_reuse, n_processed, images);
}

run_status pipeline::truncate(int32_t slot, int32_t n_keep) {
    nvtx_range nr("pipe.truncate");
    end_request(slot, "truncate");
    return l_->truncate(slot, n_keep);
}

run_status pipeline::release_slot(int32_t slot) {
    end_request(slot, "release");
    return l_->release_slot(slot);
}

bool pipeline::in_flight(int32_t slot) const {
    for (int32_t g = 0; g < 2; ++g) {
        if (fl_[g] != nullptr && std::find(members_[g].begin(), members_[g].end(), slot) != members_[g].end()) {
            return true;
        }
    }
    return false;
}

bool follower::init(engine & e, port & p, crossing & x, std::string & err) {
    e_ = &e;
    p_ = &p;
    x_ = &x;
    if (e.config().stage != 2 || int32_t(e.config().stage_buf.size()) < e.n_slots()) {
        return fail(err, "follower: the engine is not a back stage with its boundary buffers");
    }
    pre_.assign(size_t(e.n_slots()), pre());
    return true;
}

bool follower::reply(uint32_t seq, const ack & a) {
    // sent from a heap copy its send keeps: the caller's record never meets the transport
    auto          r = std::make_shared<ack>(a);
    std::string   e;
    const int32_t op = p_->send(ack_tag(seq), r.get(), sizeof(*r), r, e);
    return op >= 0 && p_->wait(op, e, ctrl_timeout_s);
}

// the members' next steps: the request's budget decides (no step: none; one token: generate; else a speculative step
// of min(d, budget - 1) drafts, its draft run now: the full-d members as one joint draft when they fit)
void follower::discard(int32_t slot) {
    if (slot < 0 || slot >= int32_t(pre_.size())) {
        return;
    }
    pre & p = pre_[size_t(slot)];
    if (p.valid && p.rounds > 0) {
        e_->split_discard_rounds(slot, p.rounds);
    }
    p.valid  = false;
    p.rounds = 0;
}

void follower::next_of(const std::vector<int32_t> & slots, std::vector<pre> & out, std::vector<std::string> & why) {
    nvtx_range nr("follower.next");
    out.assign(slots.size(), pre());
    why.assign(slots.size(), std::string());
    std::vector<uint64_t> r0(slots.size(), 0);
    for (size_t i = 0; i < slots.size(); ++i) {
        r0[i] = e_->test_target(slots[i]).round_next();
    }
    const engine_config & cf = e_->config();
    std::vector<int32_t>  lanes, alone;
    std::vector<int32_t>  dd(slots.size(), 0);
    for (size_t i = 0; i < slots.size(); ++i) {
        const int32_t s = slots[i];
        if (!e_->split_valid(s)) {
            continue;  // the slot's state is gone (a failure): the leader begins it again
        }
        target &      t = e_->test_target(s);
        const int32_t b = e_->split_budget(s);
        if (b < 1 || t.slot().u != 1) {
            continue;  // the request ended, or a prompt's handoff (the bootstrap begins it): none
        }
        if (b == 1) {
            out[i].valid = true;
            out[i].kind  = int32_t(mkind::generate);
            continue;
        }
        dd[i] = std::min(cf.spec_d, b - 1);
        (dd[i] == cf.spec_d ? lanes : alone).push_back(int32_t(i));
    }
    joint_frame * jf      = e_->split_joint();
    size_t        drafted = 0;  // the lanes the joint draft took (from the front of lanes)
    if (lanes.size() >= 2 && jf != nullptr && !joint_draft_.off) {
        target * dt[fn::xchg::max_lanes] = {};
        for (size_t q = 0; q < lanes.size() && q < size_t(fn::xchg::max_lanes); ++q) {
            dt[q] = &e_->test_target(slots[size_t(lanes[q])]);
        }
        const int32_t nm = int32_t(std::min<size_t>(lanes.size(), size_t(fn::xchg::max_lanes)));
        if (nm >= 2 && joint_frame::draft_fits(dt, nm)) {
            round_result rr[fn::xchg::max_lanes];
            bool         dok[fn::xchg::max_lanes] = {};
            std::string  dwhy[fn::xchg::max_lanes], derr;
            const auto   t0 = clk::now();
            if (jf->draft(dt, nm, cf.spec_d, rr, dok, dwhy, derr)) {
                for (int32_t q = 0; q < nm; ++q) {
                    const size_t i = size_t(lanes[size_t(q)]);
                    if (!dok[q]) {
                        why[i] = dwhy[q];
                        continue;
                    }
                    out[i].valid = true;
                    out[i].kind  = int32_t(mkind::verify);
                    out[i].d     = cf.spec_d;
                    for (int32_t j = 0; j < cf.spec_d; ++j) {
                        out[i].prop[j] = rr[q].argmax[0][j];
                    }
                    out[i].to_draft_s = since(t0);
                }
                drafted = size_t(nm);
            } else {
                joint_draft_.refuse("FNFL joint", derr, jf->capture_failed(), "these slots draft alone");
            }
        }
    }
    for (size_t q = drafted; q < lanes.size(); ++q) {
        alone.push_back(lanes[q]);
    }
    for (int32_t i : alone) {
        target &     t = e_->test_target(slots[size_t(i)]);
        round_result r;
        std::string  e;
        const auto   t0 = clk::now();
        if (!t.draft_step(dd[size_t(i)], r, e)) {
            why[size_t(i)] = "draft step: " + e;
            continue;
        }
        pre & p = out[size_t(i)];
        p.valid = true;
        p.kind  = int32_t(mkind::verify);
        p.d     = dd[size_t(i)];
        for (int32_t j = 0; j < p.d; ++j) {
            p.prop[j] = r.argmax[0][j];
        }
        p.to_draft_s = since(t0);
    }
    for (size_t i = 0; i < slots.size(); ++i) {
        if (out[i].valid) {
            out[i].rounds = e_->test_target(slots[i]).round_next() - r0[i];
        }
    }
}

void follower::rebind(engine & e) {
    e_ = &e;
    pre_.assign(size_t(e.n_slots()), pre());
    reads_.clear();
}

// the drive's tests: a forced round of rows rows on each member (2 .. 4 of them: one joint round), from the
// boundaries; each member's X-STEP, then its logits rows (both cards')
void follower::forced_group(const ctrl & c, uint32_t seq) {
    const int32_t         n    = std::clamp(c.n_members, 0, max_members);
    const int32_t         rows = n > 0 ? c.m[0].rows : 0;
    const engine_config & cf   = e_->config();
    std::vector<step_out> out(static_cast<size_t>(n));
    std::string           e;
    bool                  ok = n > 0 && rows >= 1 && rows <= fn::round::max_rows;
    std::vector<int32_t>  ops;
    for (int32_t k = 0; k < n && ok; ++k) {
        const member & m    = c.m[k];
        out[size_t(k)].slot = m.slot;
        ok = m.slot >= 0 && m.slot < e_->n_slots() && m.rows == rows && m.kind == int32_t(mkind::forced);
        for (int32_t card = 0; card < 2 && ok; ++card) {
            uint8_t * q = cf.stage_buf[size_t(m.slot)][size_t(card)];
            life::reuse(q, size_t(model::stage_bytes(rows)), "boundary buffer");
            const int32_t op =
                p_->recv(boundary_tag(c.group, seq, card, m.slot), q, size_t(model::stage_bytes(rows)), e);
            ok = op >= 0;
            if (ok) {
                ops.push_back(op);
            }
        }
    }
    // every boundary's receive waited, or (a failure) retired: none writes the slot's buffer later
    if (!ok) {
        p_->retire(ops.data(), int32_t(ops.size()));
    } else {
        ok = wait_all(*p_, ops, step_timeout_s, e);
    }
    std::vector<target *> tt;
    for (int32_t k = 0; k < n && ok; ++k) {
        tt.push_back(&e_->test_target(c.m[k].slot));
    }
    std::vector<round_result> r(static_cast<size_t>(n));
    std::vector<bool>         fine(size_t(n), ok);
    std::vector<std::string>  why(size_t(n), ok ? std::string() : "the forced group's boundaries: " + e);
    joint_frame *             jf    = e_->split_joint();
    const bool                joint = ok && n >= 2 && jf != nullptr;
    if (joint) {
        round_result rr[fn::xchg::max_lanes];
        bool         jok[fn::xchg::max_lanes] = {};
        std::string  jwhy[fn::xchg::max_lanes], jerr;
        const bool   ran = jf->forced(tt.data(), n, rows, rr, jok, jwhy, jerr);
        for (int32_t k = 0; k < n; ++k) {
            fine[size_t(k)] = ran && jok[k];
            why[size_t(k)]  = ran ? jwhy[k] : jerr;
            r[size_t(k)]    = rr[k];
        }
    } else if (ok) {
        for (int32_t k = 0; k < n; ++k) {
            std::string w;
            fine[size_t(k)] = tt[size_t(k)]->forced(rows, r[size_t(k)], w);
            why[size_t(k)]  = w;
        }
    }
    const int32_t VC = fn::head::n_vocab_card;
    for (int32_t k = 0; k < n; ++k) {
        step_out & o = out[size_t(k)];
        o.kind       = int32_t(mkind::forced);
        o.ran        = fine[size_t(k)] ? 1 : 0;
        o.joint      = joint ? 1 : 0;
        o.status     = int32_t(fine[size_t(k)] ? run_status::ok : run_status::error);
        if (!fine[size_t(k)]) {
            set_why(o.why, sizeof o.why, "forced: " + why[size_t(k)]);
        }
        auto lg = std::make_shared<std::vector<float>>(size_t(2) * size_t(rows) * size_t(VC), 0.0f);
        if (fine[size_t(k)]) {
            target & t = *tt[size_t(k)];
            for (int32_t card = 0; card < 2; ++card) {
                if (cudaSetDevice(e_->test_model().device[card]) != cudaSuccess ||
                    cudaMemcpy(lg->data() + size_t(card) * size_t(rows) * size_t(VC), t.card(card).logits,
                               size_t(rows) * size_t(VC) * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
                    o.status = int32_t(run_status::error);
                    set_why(o.why, sizeof o.why, "forced: the logits' copy");
                }
            }
        }
        // each record kept by its send until the send's end
        auto rec = std::make_shared<step_out>(o);
        outs_.push_back(rec);
        out_ops_.push_back(p_->send(step_tag(c.group, seq, c.m[k].slot), rec.get(), sizeof(step_out), rec, e));
        lgs_.push_back(lg);
        out_ops_.push_back(p_->send(logits_tag(c.group, seq, c.m[k].slot), lg->data(), lg->size() * 4, lg, e));
    }
}

void follower::step_group(const ctrl & c, uint32_t seq) {
    const int32_t n = std::clamp(c.n_members, 0, max_members);
    if (n > 0 && c.m[0].kind == int32_t(mkind::forced)) {
        forced_group(c, seq);
        return;
    }
    nvtx_range             nr("follower.step_group");
    const auto             t0 = clk::now();
    for (int32_t k = 0; k < n; ++k) {
        fn::xchg::slow::writer(c.m[k].slot, true);  // SSF1: the slow trace's gate
    }
    const engine_config &  cf = e_->config();
    std::vector<step_out>  out(static_cast<size_t>(n));
    std::vector<step_info> s(static_cast<size_t>(n));
    std::vector<int32_t>   b(size_t(n), 0);
    std::vector<bool>      ran(size_t(n), false), failed(size_t(n), false), malformed(size_t(n), false);
    std::string            e;
    // the members' boundaries
    std::vector<int32_t>   ops;
    bool                   submitted = true;
    for (int32_t k = 0; k < n; ++k) {
        const member & m     = c.m[k];
        out[size_t(k)].slot  = m.slot;
        malformed[size_t(k)] = m.slot < 0 || m.slot >= e_->n_slots() ||
                               (m.kind != int32_t(mkind::begin) && (m.rows < 1 || m.rows > fn::round::max_rows));
        if (m.kind == int32_t(mkind::begin) || malformed[size_t(k)]) {
            continue;
        }
        for (int32_t card = 0; card < 2 && submitted; ++card) {
            // the slot's buffer: its previous group's rounds ended (this follower runs them before answering)
            uint8_t * q = cf.stage_buf[size_t(m.slot)][size_t(card)];
            life::reuse(q, size_t(model::stage_bytes(m.rows)), "boundary buffer");
            const int32_t op =
                p_->recv(boundary_tag(c.group, seq, card, m.slot), q, size_t(model::stage_bytes(m.rows)), e);
            submitted = op >= 0;
            if (submitted) {
                ops.push_back(op);
            }
        }
    }
    // every boundary's receive waited, or (a failure) retired: none writes a slot's buffer later
    bool have = submitted;
    if (!submitted) {
        p_->retire(ops.data(), int32_t(ops.size()));
    } else {
        nvtx_range nw("follower.boundary_wait");
        have = wait_all(*p_, ops, step_timeout_s, e);
    }
    trace("follower", "step group " + std::to_string(c.group) + ": boundaries " + (have ? "in" : "failed: " + e));
    const auto fail_member = [&](int32_t k, run_status st, const std::string & why) {
        out[size_t(k)].status = int32_t(st);
        set_why(out[size_t(k)].why, sizeof out[size_t(k)].why, why);
        failed[size_t(k)] = true;
    };
    nvtx_phase ph("follower.enter");
    for (int32_t k = 0; k < n; ++k) {
        const member & m = c.m[k];
        if (malformed[size_t(k)]) {
            fail_member(k, run_status::bad_request,
                        "a malformed member: slot " + std::to_string(m.slot) + ", rows " + std::to_string(m.rows));
            continue;
        }
        if (!have) {
            fail_member(k, run_status::error, "the boundaries: " + e);
            continue;
        }
        const run_status st = e_->split_enter(m.slot, b[size_t(k)]);
        if (st != run_status::ok) {
            fail_member(k, st, e_->error());
            continue;
        }
        s[size_t(k)].n_before   = e_->test_target(m.slot).slot().n;
        out[size_t(k)].n_before = s[size_t(k)].n_before;
    }
    ph.next("follower.rounds");
    // the rounds: the begins (the bootstrap), the generates, the verifies (the group's joint lanes together)
    std::vector<int32_t> lanes;
    for (int32_t k = 0; k < n; ++k) {
        const member & m = c.m[k];
        if (failed[size_t(k)]) {
            continue;
        }
        target & t = e_->test_target(m.slot);
        if (m.kind == int32_t(mkind::begin)) {
            if (t.slot().u == 0) {
                std::string why;
                const bool  ok = runtime::step(t, b[size_t(k)], cf.spec_d, e_->eog(m.slot), s[size_t(k)], why);
                if (!ok) {
                    fail_member(k, run_status::error, "bootstrap: " + why);
                    continue;
                }
                ran[size_t(k)] = true;
            }
            continue;
        }
        const pre & p = pre_[size_t(m.slot)];
        if (m.kind == int32_t(mkind::generate)) {
            round_result r;
            std::string  why;
            if (!t.generate(r, why)) {
                fail_member(k, run_status::error, "generate: " + why);
                continue;
            }
            step_info & si              = s[size_t(k)];
            si.kind                     = 1;
            si.n_out                    = 1;
            si.out[0]                   = r.out[0];
            si.waits                    = 1;
            si.to_one_s                 = since(t0);
            si.end                      = is_eog(e_->eog(m.slot), r.out[0]);
            si.wall_s                   = since(t0);
            out[size_t(k)].round_n_out  = 1;
            out[size_t(k)].round_out[0] = r.out[0];
            ran[size_t(k)]              = true;
            continue;
        }
        // a verify of the proposals the back drafted at the last step's end
        bool same = p.valid && p.kind == int32_t(mkind::verify) && p.d == m.d;
        for (int32_t j = 0; j < m.d && same; ++j) {
            same = p.prop[j] == m.prop[j];
        }
        if (!same) {
            fail_member(k, run_status::error, "verify: the leader's proposals are not the back's draft");
            continue;
        }
        s[size_t(k)].kind = 2;
        s[size_t(k)].d    = m.d;
        std::copy(m.prop, m.prop + 4, s[size_t(k)].prop);
        s[size_t(k)].to_draft_s = p.to_draft_s;
        if (m.joint != 0) {
            lanes.push_back(k);
        }
    }
    std::vector<round_result> rv(static_cast<size_t>(n));
    std::vector<bool>         verified(size_t(n), false);
    joint_frame *             jf = e_->split_joint();
    if (lanes.size() >= 2 && jf != nullptr) {
        target *        tt[fn::xchg::max_lanes] = {};
        int32_t         d[fn::xchg::max_lanes] = {}, bud[fn::xchg::max_lanes] = {};
        const int32_t * prop[fn::xchg::max_lanes] = {};
        round_result    rr[fn::xchg::max_lanes];
        bool            ok[fn::xchg::max_lanes] = {};
        std::string     why[fn::xchg::max_lanes], jerr;
        for (size_t i = 0; i < lanes.size(); ++i) {
            const int32_t k = lanes[i];
            tt[i]           = &e_->test_target(c.m[k].slot);
            d[i]            = c.m[k].d;
            bud[i]          = b[size_t(k)];
            prop[i]         = c.m[k].prop;
        }
        bool       staged = false;
        const bool joint  = jf->verify(tt, int32_t(lanes.size()), d, prop, bud, rr, ok, why, staged, jerr);
        for (size_t i = 0; i < lanes.size(); ++i) {
            const int32_t k = lanes[i];
            if (!joint || !ok[i]) {
                fail_member(k, run_status::error, "joint verify: " + (joint ? why[i] : jerr));
                continue;
            }
            rv[size_t(k)]        = rr[i];
            verified[size_t(k)]  = true;
            s[size_t(k)].joint   = true;
            out[size_t(k)].joint = 1;
        }
    }
    for (int32_t k = 0; k < n; ++k) {
        const member & m = c.m[k];
        if (failed[size_t(k)] || m.kind != int32_t(mkind::verify) || verified[size_t(k)]) {
            continue;
        }
        if (m.joint != 0) {
            fail_member(k, run_status::error, "joint verify: the group's joint round did not run");
            continue;
        }
        std::string why;
        if (!e_->test_target(m.slot).verify(m.d, m.prop, b[size_t(k)], rv[size_t(k)], why)) {
            fail_member(k, run_status::error, "verify: " + why);
            continue;
        }
        verified[size_t(k)] = true;
    }
    // each verify's tokens and cut, then the catch-ups (the members' together when they fit), then each step closed
    std::vector<int32_t> cm;
    for (int32_t k = 0; k < n; ++k) {
        if (!verified[size_t(k)]) {
            continue;
        }
        target &             t = e_->test_target(c.m[k].slot);
        const round_result & r = rv[size_t(k)];
        step_out &           o = out[size_t(k)];
        o.round_n_out          = r.n_out;
        for (int32_t j = 0; j < 5; ++j) {
            o.round_out[j] = r.out[j];
        }
        std::string why;
        if (!step_cut(t, e_->eog(c.m[k].slot), r, t0, s[size_t(k)], why)) {
            fail_member(k, run_status::error, "cut: " + why);
            continue;
        }
        if (s[size_t(k)].cut) {
            o.cut      = 1;
            o.cut_rows = s[size_t(k)].n_out;
            o.n_keep   = s[size_t(k)].n_before + s[size_t(k)].n_out + 1;
        }
        cm.push_back(k);
    }
    std::vector<round_result> cr(static_cast<size_t>(n));
    std::vector<bool>         caught(size_t(n), false);
    if (cm.size() >= 2 && jf != nullptr && !joint_catchup_.off) {
        target *      ct[fn::xchg::max_lanes]   = {};
        int32_t       rows[fn::xchg::max_lanes] = {};
        const int32_t nc                        = int32_t(std::min<size_t>(cm.size(), size_t(fn::xchg::max_lanes)));
        for (int32_t q = 0; q < nc; ++q) {
            ct[q]   = &e_->test_target(c.m[cm[size_t(q)]].slot);
            rows[q] = s[size_t(cm[size_t(q)])].n_out;
        }
        if (nc >= 2 && joint_frame::catchup_fits(ct, nc)) {
            round_result rr[fn::xchg::max_lanes];
            bool         cok[fn::xchg::max_lanes] = {};
            std::string  cwhy[fn::xchg::max_lanes], cerr;
            if (jf->catchup(ct, nc, rows, rr, cok, cwhy, cerr)) {
                for (int32_t q = 0; q < nc; ++q) {
                    const int32_t k = cm[size_t(q)];
                    if (!cok[q]) {
                        fail_member(k, run_status::error, "catch-up: " + cwhy[q]);
                        continue;
                    }
                    cr[size_t(k)]     = rr[q];
                    caught[size_t(k)] = true;
                }
            } else {
                joint_catchup_.refuse("FNFL joint", cerr, jf->capture_failed(), "these slots catch up alone");
            }
        }
    }
    for (int32_t k : cm) {
        if (failed[size_t(k)]) {
            continue;
        }
        std::string why;
        if (!caught[size_t(k)] && !e_->test_target(c.m[k].slot).draft_catchup(s[size_t(k)].n_out, cr[size_t(k)], why)) {
            fail_member(k, run_status::error, "catch-up: " + why);
            continue;
        }
        step_caught(e_->eog(c.m[k].slot), cr[size_t(k)], t0, s[size_t(k)]);
        ran[size_t(k)] = true;
    }
    ph.next("follower.leave");
    // every member's step recorded (the back's request lines), then the next steps' drafts
    std::vector<int32_t> next_slots, next_idx;
    for (int32_t k = 0; k < n; ++k) {
        const member & m = c.m[k];
        step_out &     o = out[size_t(k)];
        if (failed[size_t(k)]) {
            // a member that entered its step and failed in it: the step's failure recorded (the slot lost)
            if (o.n_before >= 0 && o.status != int32_t(run_status::bad_request)) {
                bool    end  = false;
                int32_t nout = 0, tmp[5] = {};
                e_->split_leave(m.slot, false, s[size_t(k)], tmp, nout, end);
            }
            if (!malformed[size_t(k)]) {
                pre_[size_t(m.slot)].valid = false;
            }
            continue;
        }
        if (ran[size_t(k)]) {
            const step_info & si     = s[size_t(k)];
            bool              end    = false;
            int32_t           nout   = 0;
            int32_t           tmp[5] = {};
            const run_status  st     = e_->split_leave(m.slot, true, si, tmp, nout, end);
            o.ran                    = 1;
            o.status                 = int32_t(st);
            o.kind                   = si.kind;
            o.d                      = si.d;
            o.a                      = si.a;
            o.n_out                  = si.n_out;
            for (int32_t j = 0; j < 5; ++j) {
                o.out[j] = si.out[j];
            }
            o.end          = si.end ? 1 : 0;
            o.waits        = si.waits;
            o.wall_s       = si.wall_s;
            o.to_draft_s   = si.to_draft_s;
            o.to_verify_s  = si.to_verify_s;
            o.to_catchup_s = si.to_catchup_s;
            o.to_one_s     = si.to_one_s;
            o.hash         = e_->split_last_hash(m.slot);
            if (si.kind == 0) {
                o.round_n_out  = 1;
                o.round_out[0] = si.out[0];
            }
            if (st != run_status::ok) {
                set_why(o.why, sizeof o.why, e_->error());
                pre_[size_t(m.slot)].valid = false;
                continue;
            }
        }
        pre_[size_t(m.slot)].valid = false;
        if (e_->split_open(m.slot) || (m.kind == int32_t(mkind::begin) && !ran[size_t(k)])) {
            next_slots.push_back(m.slot);
            next_idx.push_back(k);
        }
    }
    ph.next("follower.drafts");
    trace("follower", "step group " + std::to_string(c.group) + ": rounds done; the next drafts");
    std::vector<pre>         nx;
    std::vector<std::string> nwhy;
    next_of(next_slots, nx, nwhy);
    trace("follower", "step group " + std::to_string(c.group) + ": answers");
    for (size_t i = 0; i < next_slots.size(); ++i) {
        step_out & o                = out[size_t(next_idx[i])];
        pre_[size_t(next_slots[i])] = nx[i];
        if (!nwhy[i].empty()) {
            // the next step's draft failed: the slot begins again from the leader (its next step unknown)
            fprintf(stderr, "FNFL split: slot %d: the next step's draft: %s\n", next_slots[i], nwhy[i].c_str());
            continue;
        }
        if (nx[i].valid) {
            o.next_kind = nx[i].kind;
            o.next_d    = nx[i].d;
            std::copy(nx[i].prop, nx[i].prop + 4, o.next_prop);
        }
    }
    // the answers (each record kept by its send until the send's end; the next command waits for them)
    ph.next("follower.answer");
    for (int32_t k = 0; k < n; ++k) {
        auto rec     = std::make_shared<step_out>(out[size_t(k)]);
        rec->back_ms = float(since(t0) * 1e3);  // APF1: the group's back stage, its input in to its answers
        outs_.push_back(rec);
        out_ops_.push_back(p_->send(step_tag(c.group, seq, c.m[k].slot), rec.get(), sizeof(step_out), rec, e));
        // SSF1: a step that ended its generation (or failed) ends the slot's writing for the trace's gate (a lone
        // slot gets no release command)
        if (rec->end != 0 || rec->status != int32_t(run_status::ok)) {
            fn::xchg::slow::writer(rec->slot, false);
        }
    }
}

void follower::read_answer(const iread & r, ack & a) {
    // MUP2 piece 4: the read's units account on this box so far (APF1: with its pacing)
    read_info ri;
    if (e_->get_read_info(r.slot, ri) && ri.units_run > 0) {
        a.acct[0]   = ri.unit_ms;
        a.acct[1]   = ri.floor_ms;
        a.acct[2]   = ri.budget_n > 0 ? ri.budget_ms_sum / ri.budget_n : -1.0;
        a.acct[3]   = ri.pause_ms;
        a.acct[4]   = ri.service_s * 1e3;
        a.acct_i[0] = static_cast<int32_t>(std::min<int64_t>(ri.units_run, INT32_MAX));
        a.acct_i[1] = ri.unit_steps;
        a.acct_i[2] = ri.most_units;
        a.acct_i[3] = ri.floor_units;
        a.acct_i[4] = ri.piece_min;
        a.acct_i[5] = ri.piece_max;
        a.acct_i[6] = ri.budget_misses;
        a.acct_i[7] = ri.below_floor ? 1 : 0;
        a.acct_i[8] = ri.budget_zero;
        a.acct_i[9] = ri.budget_unknown;
        pace_to_ack(ri.pace_ns, a);
    }
}

void follower::poll_ahead() {
    // PRD1 item 11: the next chunk's pieces posted once its header landed, when no other read's chunk is copied in
    // before it: the queue's first chunk is the same slot's, and it is that chunk (not begun) or the next is too
    const int32_t s = x_->ahead_slot();
    if (s < 0 || reads_.empty() || reads_.front().slot != s) {
        return;
    }
    read_info ri;
    e_->get_read_info(s, ri);
    const bool  next = !ri.in_chunk || (reads_.size() > 1 && reads_[1].slot == s);
    std::string e;
    if (!x_->poll(next, e)) {
        trace("follower", "crossing ahead: " + e);  // in() takes the crossing afresh and reports its failure
    }
}

void follower::read_group(bool idle) {
    if (test_hold_reads) {
        return;  // TEST: held (commands still run)
    }
    nvtx_range nr("follower.read_group");
    poll_ahead();
    iread &    r = reads_.front();
    slow_scope sl{ "follower read group" };
    sl.slot = r.slot;

    // APF1: the read work this box inserts between the writers' back stages (each group's account until its next
    // completion here: an overrun's reason), by its kind as the step ends
    struct charge {
        follower *        f;
        double            t0;
        pace::read_kind * k;

        ~charge() { f->pace_.read_ran(t0, pace::clock_ms(), *k); }
    };

    pace::read_kind kind = idle ? pace::read_kind::idle : r.lone ? pace::read_kind::small : pace::read_kind::budgeted;
    charge          ch{ this, pace::clock_ms(), &kind };
    if (r.cancel && !r.cancel_sent) {
        // the chunks the front finished before its cancel took effect (their crossings sent) run here first, then
        // the cancel: both boxes keep the same chunks
        read_info  ri;
        const bool live = e_->get_read_info(r.slot, ri) && ri.live;
        // PRD1 item 14: the back's job stops after the front's chunks, at the last one's end too (cancelled, the
        // chunk kept, no handoff: both boxes end the read alike)
        if (live) {
            e_->read_cancel_after(r.slot, r.want);
        }
        if (live && (x_->taken(r.slot) < r.want || ri.in_chunk)) {
            const read_status st =
                e_->read_step(r.slot, r.lone ? fn::n_layer : 1, r.lone ? read_mode::lone : read_mode::writers);
            if (st == read_status::yielded || st == read_status::chunk_done) {
                return;  // the next turn goes on
            }
            ack a;
            // the read ended on the back with the front's chunks: the cancel's end (PRD1 item 14: read_cancel_after
            // stops the last chunk too, so done does not come; mapped as before should it)
            a.status = int32_t(st == read_status::done ? read_status::cancelled : st);
            if (st == read_status::error) {
                set_why(a.why, sizeof a.why, e_->error());
            }
            read_answer(r, a);
            const uint32_t seq = r.seq;
            x_->retire_ahead(r.slot);  // PRD1 item 11: the read ended
            test_note('P');
            test_note(char('0' + r.slot));
            reads_.pop_front();
            reply(seq, a);
            return;
        }
        e_->read_cancel(r.slot);
        r.cancel_sent = true;
    }
    // APF1: this box's budget, its writer-pause deadline (PRD1 item 7: a small read comes as a = 0, whole; the back
    // has no small rule of its own). The idle drain is the floor, unbudgeted
    read_budget rb;
    rb.floor  = idle;
    rb.lagged = idle;  // PRD1 item 8: the idle drain launches a unit with no join (read_settle at a command)
    if (!idle && r.budgeted && !r.lone) {
        rb.pace_before         = pace_.totals();
        const pace::decision d = pace_.decide(pace::clock_ms());
        rb.budget_ms           = d.budget_ms;
        rb.known               = d.known;
        rb.pause_ms            = pace_.pause();
        rb.paced               = true;
        rb.pace_after          = pace_.totals();
    }
    read_status st = read_status::yielded;
    if (r.lone) {
        while (st == read_status::yielded || (r.cancel && st == read_status::chunk_done)) {
            st = e_->read_step(r.slot, fn::n_layer, read_mode::lone);
        }
    } else {
        st = e_->read_step(r.slot, 1, read_mode::writers, r.budgeted ? &rb : nullptr);
        if (st != read_status::error) {
            test_note('L');
        }
        read_info ri;
        if (!idle && r.budgeted && e_->get_read_info(r.slot, ri) && ri.last_minimum) {
            kind = pace::read_kind::minimum;  // only the progress unit ran, already past the budget
        }
        if (st == read_status::yielded || (r.cancel && st == read_status::chunk_done)) {
            return;  // the chunk (or the cancel's) goes on at the next turn
        }
    }
    ack a;
    a.status = int32_t(st);
    if (st == read_status::error) {
        set_why(a.why, sizeof a.why, e_->error());
    }
    read_answer(r, a);
    const uint32_t seq = r.seq;
    if (st != read_status::chunk_done) {
        x_->retire_ahead(r.slot);  // PRD1 item 11: the read ended
        test_note('P');
        test_note(char('0' + r.slot));
    }
    reads_.pop_front();
    reply(seq, a);
}

void follower::test_note(char c) {
    if (test_order) {
        std::lock_guard<std::mutex> g(test_mu_);
        test_log_ += c;
    }
}

std::string follower::test_log() const {
    std::lock_guard<std::mutex> g(test_mu_);
    return test_log_;
}

bool follower::read_settle() {
    if (!e_->read_draining()) {
        return true;
    }
    const double t0    = pace::clock_ms();
    test_in_flight_max = std::max(test_in_flight_max.load(), e_->read_in_flight());
    ++test_settles;
    const bool ok = e_->read_settle();
    pace_.read_ran(t0, pace::clock_ms(), pace::read_kind::idle);  // APF1: the drain's join, before the command
    if (!ok && !reads_.empty()) {
        iread & r = reads_.front();
        ack     a;
        a.status = int32_t(read_status::error);
        set_why(a.why, sizeof a.why, e_->error());
        read_answer(r, a);
        const uint32_t seq = r.seq;
        x_->retire_ahead(r.slot);
        test_note('P');
        test_note(char('0' + r.slot));
        reads_.pop_front();
        reply(seq, a);
    }
    return ok;
}

bool follower::serve(std::string & why) {
    uint32_t             seq = 1;
    std::vector<int32_t> tokens;
    for (;;) {
        // the previous answers' sends end before their records go (a failure retires the rest first)
        {
            nvtx_range  nw("follower.out_wait");
            std::string e;
            if (!wait_all(*p_, out_ops_, step_timeout_s, e)) {
                out_ops_.clear();
                why = e;
                return false;
            }
        }
        out_ops_.clear();
        outs_.clear();
        lgs_.clear();
        // the command's record on the heap, kept by its receive (a given-up wait whose receive matched still lands
        // in it, never in a reused frame)
        auto        cr = std::make_shared<ctrl>();
        ctrl &      c  = *cr;
        std::string e;
        bool        got = false;
        int32_t     cop = -1;
        nvtx_phase  ph("follower.cmd_wait");  // then the command (until the end of this turn)
        while (!got) {
            if (cop < 0) {
                cop = p_->recv(ctrl_tag(seq, 0), cr.get(), sizeof(c), cr, e);
                if (cop < 0) {
                    why = e;
                    return false;
                }
            }
            const bool waiting = !reads_.empty();
            bool       done    = false;
            if (waiting && !test_hold_reads && !p_->test(cop, done, e)) {
                why = e;
                return false;
            }
            // APF1 (pace::back_choice, host-checked): a command that came runs first; with none, one idle-drain unit
            // while reads wait; else a blocking wait for the next command
            const pace::back_next nx = pace::back_choice(done, waiting, test_hold_reads);
            if (nx == pace::back_next::command) {
                got = true;
                read_settle();  // PRD1 item 8: the drain's units joined before the command, charged to the read
                if (c.op == int32_t(op::step_group)) {
                    test_note('C');
                }
                continue;
            }
            if (nx == pace::back_next::idle_unit) {
                if (p_->lost()) {
                    why = "the leader was lost";
                    return false;
                }
                read_group(true);  // MUP2: no command waits (the floor): one unit
                continue;
            }
            got = p_->wait(cop, e, idle_poll_s);
            cop = -1;  // finished either way (a timed-out wait abandons it)
            if (!got && p_->lost()) {
                why = e;
                return false;
            }
            if (!got) {
                // SSF1: a whole idle wait with no command: no slot is writing now (a request can end with no slot
                // command, e.g. a lone slot's stop string), so a stale gate prints at most this one line
                fn::xchg::slow::set_writers(0);
            }
        }
        ph.next("follower.command");
        slow_scope sc{ "follower command" };
        sc.seq   = seq;
        sc.op    = c.op;
        sc.slot  = c.slot;
        sc.group = c.op == int32_t(op::step_group) ? c.group : -1;
        trace("follower",
              "command " + std::to_string(seq) + " op " + std::to_string(c.op) + " slot " + std::to_string(c.slot));
        if (c.n_tokens > 0) {
            auto          tr = std::make_shared<std::vector<int32_t>>(size_t(c.n_tokens), 0);  // kept by its receive
            const int32_t op = p_->recv(ctrl_tag(seq, 1), tr->data(), size_t(c.n_tokens) * 4, tr, e);
            if (op < 0 || !p_->wait(op, e, ctrl_timeout_s)) {
                why = e;
                return false;
            }
            tokens = *tr;
        }
        ack        a;
        const auto status = [&](run_status st) {
            a.status = int32_t(st);
            if (st != run_status::ok) {
                set_why(a.why, sizeof a.why, e_->error());
            }
        };
        const op   o = static_cast<op>(c.op);
        // SSF1: the slow trace's gate: a slot generates from its first step group to its next request's command
        if (o == op::set_sampling || o == op::prefill || o == op::read_begin || o == op::truncate ||
            o == op::release_slot || o == op::test_prompt || o == op::pos_offset) {
            fn::xchg::slow::writer(c.slot, false);
            pace_.slot_ended(c.slot);  // APF1: the slot's next request is a new member
        }
        // a step group names its slots in its members (checked by split_enter); the others name one, or none
        const bool slotless = o == op::pool || o == op::bye || o == op::set_draft || o == op::step_group ||
                              o == op::reconfigure || o == op::test_route;
        if (c.slot >= e_->n_slots() || (c.slot < 0 && !slotless)) {
            a.status = int32_t(run_status::bad_request);
            set_why(a.why, sizeof a.why, "slot " + std::to_string(c.slot));
        } else {
            switch (o) {
                case op::set_sampling:
                    {
                        request_options ro;
                        ro.sampling       = e_->test_target(c.slot).sampling();
                        ro.sampling.seed  = c.seed;
                        ro.sampling.temp  = c.temp;
                        ro.sampling.top_k = c.top_k;
                        ro.sampling.top_p = c.top_p;
                        ro.sampling.min_p = c.min_p;
                        ro.max_tokens     = c.max_tokens;
                        ro.n_ctx          = c.n_ctx;
                        ro.ignore_eos     = c.ignore_eos != 0;
                        discard(c.slot);
                        status(e_->set_sampling(c.slot, ro));
                        break;
                    }
                case op::prefill:
                    {
                        int32_t                   np = 0, nt = 0;
                        std::vector<prompt_image> im;  // VIS1: the spans after the tokens
                        if (c.graphs >= 0) {
                            model::prompt_graphs(c.graphs != 0);  // APF2: the front's switch
                        }
                        x_->expect(c.slot, c.base);
                        discard(c.slot);
                        if (!payload_images(tokens, c.n_tokens, c.b, nt, im)) {
                            a.status = int32_t(run_status::bad_request);
                            set_why(a.why, sizeof a.why, "prefill: the command's images do not add up");
                            break;
                        }
                        status(e_->prefill(c.slot, tokens.data(), nt, c.a, np, {}, im.empty() ? nullptr : &im));
                        a.a = np;
                        break;
                    }
                case op::read_begin:
                    {
                        int32_t                   np = 0, nt = 0;
                        std::vector<prompt_image> im;  // VIS1: the spans after the tokens
                        if (c.graphs >= 0) {
                            model::prompt_graphs(c.graphs != 0);  // APF2: the front's switch
                        }
                        x_->expect(c.slot, c.base);
                        discard(c.slot);
                        if (!payload_images(tokens, c.n_tokens, c.b, nt, im)) {
                            a.status = -1;
                            set_why(a.why, sizeof a.why, "read_begin: the command's images do not add up");
                            break;
                        }
                        a.status = e_->read_begin(c.slot, tokens.data(), nt, c.a, np, im.empty() ? nullptr : &im);
                        a.a      = np;
                        if (a.status != 0) {
                            set_why(a.why, sizeof a.why, e_->error());
                        } else {
                            read_info ri;  // PRD1 item 11: the read's plan, for the crossings taken ahead
                            e_->get_read_info(c.slot, ri);
                            x_->plan(c.slot, ri.chunks);
                        }
                        break;
                    }
                case op::read_step:
                    {
                        // a back chunk (c.a 0: whole, in the lone geometry; 1: interleaved), queued behind the earlier
                        // ones and answered at its end (read_group)
                        iread r;
                        r.slot     = c.slot;
                        r.seq      = seq;
                        r.lone     = c.a == 0;
                        r.cancel   = c.b != 0;
                        r.want     = c.c;
                        r.budgeted = c.a == 2;  // MUP2: units under this box's budget (APF1: its deadline)
                        if (r.budgeted && pace::pause_valid(int64_t(c.pause_ms))) {
                            pace_.set_pause(int32_t(c.pause_ms));  // APF1: the next decision takes it
                        }
                        reads_.push_back(r);
                        if (r.lone && reads_.size() == 1) {
                            read_group();  // a lone chunk with nothing before it: at once
                        }
                        ++seq;
                        continue;
                    }
                case op::read_cancel:
                    {
                        e_->read_cancel(c.slot);
                        read_status st = read_status::yielded;
                        while (st == read_status::yielded || st == read_status::chunk_done) {
                            st = e_->read_step(c.slot, fn::n_layer, read_mode::lone);
                        }
                        a.status = int32_t(st);
                        break;
                    }
                case op::truncate:
                    {
                        // TTTS8: the checkpoints change on both boxes together (leader::truncate)
                        discard(c.slot);
                        prefiller &   pf  = e_->prefill_driver();
                        const int32_t had = pf.book().count_owner(c.slot);
                        status(e_->truncate(c.slot, c.a));
                        if (c.slot >= 0 && test_fail_cut.load() == c.slot) {
                            test_fail_cut.store(-1);
                            pf.drop_owner(c.slot);
                            e_->to_restore(c.slot);
                            fprintf(stderr, "FNFL split: slot %d: truncate to %d: TEST: the follower's cut failed\n",
                                    c.slot, c.a);
                        }
                        a.b = had > 0 && pf.book().count_owner(c.slot) == 0 ? 1 : 0;
                        if (c.c != 0 && a.b == 0 && had > 0) {
                            pf.drop_owner(c.slot);
                            fprintf(
                                stderr,
                                "FNFL split: slot %d: truncate to %d: the leader dropped the slot's checkpoints; the "
                                "follower drops its %d too\n",
                                c.slot, c.a, had);
                        }
                        if (c.b == 0) {
                            e_->to_restore(c.slot);
                        }
                        a.a = e_->in_place(c.slot) ? 1 : 0;
                        break;
                    }
                case op::release_slot:
                    discard(c.slot);
                    status(e_->release_slot(c.slot));
                    break;
                case op::pool:
                    {
                        pool_info pi;
                        a.status = e_->get_pool_info(pi) ? 0 : 1;
                        a.a      = pi.pages;
                        a.b      = pi.free_pages;
                        a.c      = pi.page_cells;
                        a.x      = pi.capacity_cells;
                        break;
                    }
                case op::step_group:
                    {
                        // APF1: the group's back stage on this box's book: its input in (the command, its drain
                        // joined), its answers sent; P, the front's groups and the front's service carried with it.
                        // The drive's forced groups are no writers
                        const bool    forced = c.n_members > 0 && c.m[0].kind == int32_t(mkind::forced);
                        const double  t0     = pace::clock_ms();
                        const int32_t n      = std::clamp(c.n_members, 0, max_members);
                        if (!forced) {
                            if (pace::pause_valid(int64_t(c.pause_ms))) {
                                pace_.set_pause(int32_t(c.pause_ms));
                            }
                            pace_.live_groups(c.live, c.gen, t0);
                        }
                        step_group(c, seq);
                        if (!forced && c.group >= 0 && c.group < 2) {
                            int32_t ms[max_members] = {};
                            for (int32_t k = 0; k < n; ++k) {
                                ms[k] = c.m[k].slot;
                            }
                            pace_.back_ran(c.group, c.gen[c.group], ms, n, t0, pace::clock_ms(), c.front_ms);
                        }
                    }
                    ++seq;
                    test_note('Q');
                    if (!reads_.empty()) {
                        read_group();  // read units after each step group's back half, under this box's budget
                    }
                    continue;          // answered by X-STEP
                case op::drain:
                    {
                        std::string de;
                        a.status = x_->drop(c.slot, c.base, c.a, de) ? 0 : 1;
                        if (a.status != 0) {
                            set_why(a.why, sizeof a.why, de);
                        }
                        break;
                    }
                case op::pos_offset:
                    discard(c.slot);  // MUP1: an early draft at the old positions is not used
                    status(e_->set_pos_offset(c.slot, c.a));
                    break;
                case op::set_draft:
                    for (int32_t sl = 0; sl < int32_t(pre_.size()); ++sl) {
                        discard(sl);
                    }
                    status(e_->set_draft(c.a, c.b != 0));
                    break;
                case op::probe:
                    {
                        std::string pe;
                        probe_out_.clear();
                        const bool pok = probe_fn && probe_fn(c.slot, c.a, probe_out_, pe);
                        a.status       = int32_t(pok ? run_status::ok : run_status::error);
                        a.a            = pok ? int32_t(probe_out_.size()) : -1;
                        if (!pok) {
                            set_why(a.why, sizeof a.why, probe_fn ? pe : "no probe on this follower");
                        }
                        if (!reply(seq, a)) {
                            why = "the acknowledgment of command " + std::to_string(seq);
                            return false;
                        }
                        if (pok && !probe_out_.empty()) {
                            // sent from a heap copy its send keeps
                            auto pr = std::make_shared<std::vector<uint8_t>>(probe_out_);
                            out_ops_.push_back(p_->send(probe_tag(seq), pr->data(), pr->size(), pr, e));
                        }
                        ++seq;
                        continue;
                    }
                case op::reconfigure:
                    {
                        std::string re;
                        engine *    ne = reconfigure_fn ? reconfigure_fn(c, re) : nullptr;
                        if (ne != nullptr) {
                            rebind(*ne);
                        }
                        a.status = int32_t(ne != nullptr ? run_status::ok : run_status::error);
                        if (ne == nullptr) {
                            set_why(a.why, sizeof a.why, reconfigure_fn ? re : "no reconfigure on this follower");
                        }
                        break;
                    }
                case op::test_route:
                    {
                        model::prompt_route_force f;
                        f.dense  = c.a;
                        f.hc     = c.b;
                        f.expert = c.c;
                        f.graph  = static_cast<int32_t>(c.base);
                        e_->prefill_driver().test_force_route(f);
                        model::prompt_route_collect(true);
                        model::prompt_path::route_counts rc;
                        uint32_t                         rungs = 0;
                        model::prompt_route_totals(rc, rungs);
                        for (int32_t i = 0; i < route_count_fields; ++i) {
                            a.acct_i[i] = static_cast<int32_t>(rc.*route_fields[i]);
                        }
                        break;
                    }
                case op::test_prompt:
                    {
                        target &    t = e_->test_target(c.slot);
                        std::string te;
                        discard(c.slot);
                        const std::vector<int32_t> in  = c.n_tokens > 0 ? tokens : std::vector<int32_t>();
                        // MUP2 (a 1): the processed tokens kept, the tokens appended
                        const bool                 tok = c.a == 1 ? t.keep_processed(te) : t.reset(te);
                        if (tok && c.a == 1) {
                            t.append(in);
                        } else if (tok) {
                            t.set_prompt(in);
                        }
                        a.status = int32_t(tok ? run_status::ok : run_status::error);
                        if (!tok) {
                            set_why(a.why, sizeof a.why, te);
                        }
                        break;
                    }
                case op::bye:
                    reply(seq, a);
                    return true;
                default:
                    a.status = int32_t(run_status::bad_request);
                    set_why(a.why, sizeof a.why, "op " + std::to_string(c.op));
                    break;
            }
        }
        if (!reply(seq, a)) {
            why = "the acknowledgment of command " + std::to_string(seq);
            return false;
        }
        ++seq;
    }
}

}  // namespace fn::runtime::stage
