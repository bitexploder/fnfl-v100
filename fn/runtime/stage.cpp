#include "fn/runtime/stage.h"

#include "fn/kernels/head/head.h"
#include "fn/model/pass.h"
#include "fn/model/prompt.h"
#include "fn/runtime/joint.h"

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

namespace lk = fn::xchg::link;
using clk    = std::chrono::steady_clock;

constexpr double  ctrl_timeout_s = 60.0;   // a command's answer (a prefill's: crossing timeout)
constexpr double  step_timeout_s = 120.0;  // a step group's X-STEP (back half, catch-up, next draft)
constexpr int32_t max_b          = fn::round::max_rows;

double since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

// FNFL_SPLIT_TRACE=1: one stderr line a command and a step group's phase on either side (the protocol's order)
void trace(const char * who, const std::string & what) {
    static const bool on = getenv("FNFL_SPLIT_TRACE") != nullptr;
    if (on) {
        fprintf(stderr, "FNFL split trace: %s: %s\n", who, what.c_str());
        fflush(stderr);
    }
}

bool fail(std::string & err, const std::string & what) {
    err = what;
    return false;
}

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

bool is_eog(const std::vector<int32_t> & eog, int32_t tok) {
    return std::find(eog.begin(), eog.end(), tok) != eog.end();
}

// ---- the link's port ----

class link_port_impl : public port {
  public:
    explicit link_port_impl(lk::link & l) : l_(l) {}

    int32_t send(uint64_t tag, const void * p, size_t n, std::string & err) override { return l_.send(tag, p, n, err); }

    int32_t recv(uint64_t tag, void * p, size_t n, std::string & err) override { return l_.recv(tag, p, n, err); }

    bool wait(int32_t op, std::string & err, double timeout_s) override { return l_.wait(op, err, timeout_s); }

    bool test(int32_t op, bool & done, std::string & err) override { return l_.test(op, done, err); }

    bool map(void * p, size_t n, std::string & err) override { return l_.map(p, n, err); }

    bool lost() const override { return l_.lost(); }

  private:
    lk::link & l_;
};

// ---- split-test's in-process pair ----

struct loop_hub {
    std::mutex              mu;
    bool                    closed = false;
    std::condition_variable cv;

    struct parked {
        uint64_t             tag;
        std::vector<uint8_t> data;
    };

    struct rx {
        uint64_t    tag  = 0;
        void *      p    = nullptr;
        size_t      n    = 0;
        bool        done = false, ok = false;
        std::string why;
    };

    std::deque<parked>    unexpected[2];  // sent to side s, not yet received
    std::map<int32_t, rx> ops[2];         // side s's operations (sends complete at once)
    std::vector<int32_t>  order[2];       // side s's posted receives, in post order
    int32_t               next = 1;
};

class loop_port : public port {
  public:
    loop_port(std::shared_ptr<loop_hub> h, int32_t side) : h_(std::move(h)), s_(side) {}

    int32_t send(uint64_t tag, const void * p, size_t n, std::string & err) override {
        (void) err;
        std::lock_guard<std::mutex> g(h_->mu);
        const int32_t               id = h_->next++;
        loop_hub::rx &              me = h_->ops[s_][id];
        me.done = me.ok    = true;
        const int32_t peer = 1 - s_;
        for (size_t i = 0; i < h_->order[peer].size(); ++i) {
            loop_hub::rx & r = h_->ops[peer][h_->order[peer][i]];
            if (!r.done && r.tag == tag) {
                if (n == r.n) {
                    std::memcpy(r.p, p, n);
                    r.ok = true;
                } else {
                    r.why = "refused: " + lk::tag_str(tag) + ": received " + std::to_string(n) + " bytes, expected " +
                            std::to_string(r.n);
                }
                r.done = true;
                h_->order[peer].erase(h_->order[peer].begin() + int64_t(i));
                h_->cv.notify_all();
                return id;
            }
        }
        h_->unexpected[peer].push_back(
            { tag, std::vector<uint8_t>(static_cast<const uint8_t *>(p), static_cast<const uint8_t *>(p) + n) });
        return id;
    }

    int32_t recv(uint64_t tag, void * p, size_t n, std::string & err) override {
        (void) err;
        std::lock_guard<std::mutex> g(h_->mu);
        const int32_t               id = h_->next++;
        loop_hub::rx &              r  = h_->ops[s_][id];
        r.tag                          = tag;
        r.p                            = p;
        r.n                            = n;
        for (auto it = h_->unexpected[s_].begin(); it != h_->unexpected[s_].end(); ++it) {
            if (it->tag == tag) {
                if (it->data.size() == n) {
                    std::memcpy(p, it->data.data(), n);
                    r.ok = true;
                } else {
                    r.why = "refused: " + lk::tag_str(tag) + ": received " + std::to_string(it->data.size()) +
                            " bytes, expected " + std::to_string(n);
                }
                r.done = true;
                h_->unexpected[s_].erase(it);
                return id;
            }
        }
        h_->order[s_].push_back(id);
        return id;
    }

    bool wait(int32_t op, std::string & err, double timeout_s) override {
        std::unique_lock<std::mutex> g(h_->mu);
        auto                         it = h_->ops[s_].find(op);
        if (it == h_->ops[s_].end()) {
            return fail(err, "loop port: no operation " + std::to_string(op));
        }
        const bool done =
            h_->cv.wait_for(g, std::chrono::duration<double>(timeout_s), [&]() { return it->second.done; });
        if (!done) {
            // abandoned: a later send for its tag is parked, not delivered into this buffer
            auto & ord = h_->order[s_];
            ord.erase(std::remove(ord.begin(), ord.end(), op), ord.end());
            const uint64_t tag = it->second.tag;
            h_->ops[s_].erase(it);
            return fail(err, "no completion in " + std::to_string(timeout_s) + " s: " + lk::tag_str(tag));
        }
        const bool ok = it->second.ok;
        if (!ok) {
            err = it->second.why;
        }
        h_->ops[s_].erase(it);
        return ok;
    }

    bool test(int32_t op, bool & done, std::string & err) override {
        done = false;
        {
            std::lock_guard<std::mutex> g(h_->mu);
            auto                        it = h_->ops[s_].find(op);
            if (it == h_->ops[s_].end()) {
                return fail(err, "loop port: no operation " + std::to_string(op));
            }
            if (!it->second.done) {
                return true;
            }
        }
        done = true;
        return wait(op, err, 0.0);
    }

    bool map(void *, size_t, std::string &) override { return true; }

    bool lost() const override {
        std::lock_guard<std::mutex> g(h_->mu);
        return h_->closed;
    }

    void close() override {
        std::lock_guard<std::mutex> g(h_->mu);
        h_->closed = true;
        h_->cv.notify_all();
    }

  private:
    std::shared_ptr<loop_hub> h_;
    int32_t                   s_;
};

uint64_t ctrl_tag(uint32_t seq, uint8_t part) {
    return lk::make_tag(lk::kind::ctrl, part, seq, 0);
}

uint64_t ack_tag(uint32_t seq) {
    return lk::make_tag(lk::kind::ack, 0, seq, 0);
}

uint64_t boundary_tag(int32_t group, uint32_t seq, int32_t card, int32_t slot) {
    return lk::make_tag(lk::kind::boundary, uint8_t(group), seq, uint8_t(card), uint8_t(slot));
}

uint64_t step_tag(int32_t group, uint32_t seq, int32_t slot) {
    return lk::make_tag(lk::kind::step, uint8_t(group), seq, 0, uint8_t(slot));
}

// a forced member's back logits (the drive's tests), and a probe's answer bytes
uint64_t logits_tag(int32_t group, uint32_t seq, int32_t slot) {
    return lk::make_tag(lk::kind::step, uint8_t(group), seq, 1, uint8_t(slot));
}

uint64_t probe_tag(uint32_t seq) {
    return lk::make_tag(lk::kind::ack, 1, seq, 0);
}

uint64_t crossing_tag(int64_t n, int32_t piece) {
    return lk::make_tag(lk::kind::prompt, 0, uint32_t(n), 0, uint8_t(piece));
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

std::unique_ptr<port> link_port(lk::link & l) {
    return std::make_unique<link_port_impl>(l);
}

void loop_pair(std::unique_ptr<port> & a, std::unique_ptr<port> & b) {
    auto h = std::make_shared<loop_hub>();
    a      = std::make_unique<loop_port>(h, 0);
    b      = std::make_unique<loop_port>(h, 1);
}

// ---- the boundary buffers ----

buffers::~buffers() {
    for (auto & s : slot) {
        for (uint8_t * p : s) {
            if (p != nullptr) {
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
            slot[size_t(s)][size_t(c)] = static_cast<uint8_t *>(q);
            if (p != nullptr && !p->map(q, n, err)) {
                return false;
            }
        }
    }
    return true;
}

// ---- the prompt crossing ----

bool crossing::init(port & p, bool front, int32_t k, std::string & err) {
    p_     = &p;
    front_ = front;
    k_     = k;
    model::prompt_chunk big;
    big.rows    = model::prompt_chunk_rows;
    big.half[0] = model::prompt_half_rows;
    big.half[1] = model::prompt_chunk_rows - model::prompt_half_rows;
    cap_        = model::prompt_path::crossing_bytes(big);
    for (int32_t b = 0; b < 2; ++b) {
        void * q = nullptr;
        if (!cuda_ok(cudaHostAlloc(&q, size_t(cap_), cudaHostAllocPortable), "crossing buffer", err) ||
            !p.map(q, size_t(cap_), err)) {
            return false;
        }
        buf_[b] = static_cast<uint8_t *>(q);
    }
    return true;
}

crossing::~crossing() {
    for (int32_t b = 0; b < 2; ++b) {
        if (buf_[b] != nullptr) {
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
        base_[slot]  = base;
        sent_[slot]  = 0;
        taken_[slot] = base;
    }
}

bool crossing::drop(int32_t slot, int64_t base, int64_t count, std::string & err) {
    if (front_ || slot < 0 || slot >= max_split_slots) {
        return fail(err, "crossing drop: the follower's, a slot");
    }
    std::vector<uint8_t> tmp;
    for (int64_t n = std::max(taken_[slot], base); n < base + count; ++n) {
        crossing_head h;
        const int32_t hop = p_->recv(crossing_tag(n, 255), &h, sizeof(h), err);
        if (hop < 0 || !p_->wait(hop, err, timeout_s)) {
            return false;
        }
        tmp.resize(size_t(std::min<int64_t>(h.bytes, piece_bytes)));
        for (int32_t i = 0; i < h.pieces; ++i) {
            const int64_t at = int64_t(i) * piece_bytes;
            const int32_t op =
                p_->recv(crossing_tag(n, i), tmp.data(), size_t(std::min<int64_t>(piece_bytes, h.bytes - at)), err);
            if (op < 0 || !p_->wait(op, err, timeout_s)) {
                return false;
            }
        }
        taken_[slot] = n + 1;
    }
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
    // release: the buffer's previous sends completed
    for (int32_t op : ops_[b]) {
        if (!p_->wait(op, err, timeout_s)) {
            return false;
        }
    }
    ops_[b].clear();
    if (!pp.crossing_out(c, k_, buf_[b], err)) {
        return false;
    }
    for (int32_t q = 0; q < 2; ++q) {
        if (!cuda_ok(cudaSetDevice(t.card(q).c.device), "device", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(q))), "crossing out", err)) {
            return false;
        }
    }
    crossing_head & h = head_;
    h                 = crossing_head();
    h.slot            = slot;
    h.rows            = c.rows;
    h.half[0]         = c.half[0];
    h.half[1]         = c.half[1];
    h.start           = c.start;
    h.seq             = c.seq;
    h.bytes           = bytes;
    h.pieces          = static_cast<int32_t>((bytes + piece_bytes - 1) / piece_bytes);
    // the header first, waited (head_ is the next chunk's too), then the pieces from the pinned buffer
    const int32_t hop = p_->send(crossing_tag(n, 255), &head_, sizeof(head_), err);
    if (hop < 0 || !p_->wait(hop, err, timeout_s)) {
        return false;
    }
    for (int32_t i = 0; i < h.pieces; ++i) {
        const int64_t at = int64_t(i) * piece_bytes;
        const int32_t op =
            p_->send(crossing_tag(n, i), buf_[b] + at, size_t(std::min<int64_t>(piece_bytes, bytes - at)), err);
        if (op < 0) {
            return false;
        }
        ops_[b].push_back(op);
    }
    sent_[slot] = std::max<int64_t>(sent_[slot], c.index + 1);
    ++n_;
    bytes_ += bytes;
    return true;
}

bool crossing::drain(std::string & err) {
    for (int32_t b = 0; b < 2; ++b) {
        for (int32_t op : ops_[b]) {
            if (!p_->wait(op, err, timeout_s)) {
                return false;
            }
        }
        ops_[b].clear();
    }
    return true;
}

bool crossing::in(target & t, model::prompt_path & pp, model::prompt_chunk & c, std::string & err) {
    if (front_) {
        return fail(err, "crossing: the leader receives none");
    }
    const int32_t slot = t.slot_index();
    const int64_t n    = base_[slot] + c.index;
    const int32_t b    = static_cast<int32_t>(n & 1);
    // the buffer's previous copy-in completed on both cards
    for (int32_t q = 0; q < 2 && used_[b]; ++q) {
        if (!cuda_ok(cudaSetDevice(t.card(q).c.device), "device", err) ||
            !cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(done_[b][q])), "crossing buffer", err)) {
            return false;
        }
    }
    crossing_head h;
    const int32_t hop = p_->recv(crossing_tag(n, 255), &h, sizeof(h), err);
    if (hop < 0 || !p_->wait(hop, err, timeout_s)) {
        return false;
    }
    taken_[slot]     = n;  // the head is taken: a mismatch below still takes the pieces, then fails
    const bool match = h.slot == slot && h.start == c.start && h.rows == c.rows && h.half[0] == c.half[0] &&
                       h.half[1] == c.half[1] && h.bytes == model::prompt_path::crossing_bytes(c) && h.bytes <= cap_;
    if (!match) {
        std::vector<uint8_t> tmp(size_t(std::min<int64_t>(std::max<int64_t>(h.bytes, 0), piece_bytes)));
        for (int32_t i = 0; i < h.pieces && h.bytes > 0; ++i) {
            const int64_t at = int64_t(i) * piece_bytes;
            std::string   e2;
            const int32_t op =
                p_->recv(crossing_tag(n, i), tmp.data(), size_t(std::min<int64_t>(piece_bytes, h.bytes - at)), e2);
            if (op < 0 || !p_->wait(op, e2, timeout_s)) {
                break;
            }
        }
        taken_[slot] = n + 1;
        return fail(err, "crossing " + std::to_string(n) + ": the leader's chunk (slot " + std::to_string(h.slot) +
                             ", " + std::to_string(h.start) + "+" + std::to_string(h.rows) +
                             ") is not this one (slot " + std::to_string(slot) + ", " + std::to_string(c.start) + "+" +
                             std::to_string(c.rows) + ")");
    }
    std::vector<int32_t> ops;
    for (int32_t i = 0; i < h.pieces; ++i) {
        const int64_t at = int64_t(i) * piece_bytes;
        const int32_t op =
            p_->recv(crossing_tag(n, i), buf_[b] + at, size_t(std::min<int64_t>(piece_bytes, h.bytes - at)), err);
        if (op < 0) {
            return false;
        }
        ops.push_back(op);
    }
    for (int32_t op : ops) {
        if (!p_->wait(op, err, timeout_s)) {
            return false;
        }
    }
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
                           const std::vector<int32_t> & eog) {
    std::string s = "k=" + std::to_string(k) + " slots=" + std::to_string(n_slots) + " cells=" + std::to_string(cells) +
                    " pool=" + std::to_string(pool_cells) + " kv=" + (kv_q8 ? "q8" : "bf16") +
                    " d=" + std::to_string(spec_d) + " att=" + attention + " eog=";
    std::vector<int32_t> e = eog;
    std::sort(e.begin(), e.end());
    for (size_t i = 0; i < e.size(); ++i) {
        s += (i ? "," : "") + std::to_string(e[i]);
    }
    return s;
}

std::string hello_hash(const std::string & text) {
    fnv128 h;
    h.add(text.data(), text.size());
    return h.hex();
}

// ---- the leader ----

struct leader::pending {
    int32_t              group = 0;
    uint32_t             seq   = 0;
    int32_t              n     = 0;
    member               m[max_members];
    int32_t              b[max_members]  = {};
    run_status           st[max_members] = {};
    std::string          why[max_members];
    target::front_round  fr[max_members];
    bool                 fronted[max_members] = {};
    step_out             out[max_members];
    int32_t              rops[max_members] = {};
    std::vector<int32_t> sops;
    bool                 sent = false;
    clk::time_point      t0;
    ctrl                 c;  // the group's command: its send reads it until group_end waits for the send
    int32_t              lops[max_members] = { -1, -1, -1, -1, -1, -1, -1, -1 };  // forced members' logits receives
    std::vector<float>   lg[max_members];                                         // and their buffers
};

bool leader::init(engine & e, port & p, crossing & x, std::string & err) {
    e_ = &e;
    p_ = &p;
    x_ = &x;
    if (e.config().stage != 1 || int32_t(e.config().stage_buf.size()) < e.n_slots()) {
        return fail(err, "leader: the engine is not a front stage with its boundary buffers");
    }
    next_.assign(size_t(e.n_slots()), next_step());
    read_acks_.assign(size_t(e.n_slots()), {});
    read_base_.assign(size_t(e.n_slots()), 0);
    read_end_.assign(size_t(e.n_slots()), -1);
    logits_.assign(size_t(e.n_slots()), {});
    return true;
}

void leader::drain_read(int32_t slot, int64_t base) {
    if (lost_) {
        return;
    }
    std::string e;
    if (!x_->drain(e)) {
        lose(e);
        return;
    }
    ctrl c;
    c.op   = int32_t(op::drain);
    c.slot = slot;
    c.base = base;
    c.a    = static_cast<int32_t>(x_->sent(slot));
    ack a;
    if (command(c, nullptr, a, 3600.0) && a.status != 0) {
        lose("the follower could not drop a failed read's crossings: " + why_of(a.why, sizeof a.why));
    }
}

void leader::lose(const std::string & why) {
    if (!lost_) {
        lost_ = true;
        why_  = why.rfind("split stage lost", 0) == 0 ? why : "split stage lost: " + why;
        fprintf(stderr, "FNFL split: %s; every later request is refused until the leader restarts\n", why_.c_str());
        fflush(stderr);
    }
}

bool leader::send_ctrl(ctrl & c, const int32_t * tokens, int32_t & ack_op, ack & a) {
    if (lost_) {
        err_ = why_;
        return false;
    }
    const uint32_t seq = seq_++;
    trace("leader",
          "command " + std::to_string(seq) + " op " + std::to_string(c.op) + " slot " + std::to_string(c.slot));
    std::string e;
    ack_op             = p_->recv(ack_tag(seq), &a, sizeof(a), e);
    const int32_t c_op = ack_op < 0 ? -1 : p_->send(ctrl_tag(seq, 0), &c, sizeof(c), e);
    const int32_t t_op =
        c_op < 0 || c.n_tokens <= 0 ? 0 : p_->send(ctrl_tag(seq, 1), tokens, size_t(c.n_tokens) * 4, e);
    if (ack_op < 0 || c_op < 0 || t_op < 0 || !p_->wait(c_op, e, ctrl_timeout_s) ||
        (c.n_tokens > 0 && !p_->wait(t_op, e, ctrl_timeout_s))) {
        lose(e);
        err_ = why_;
        return false;
    }
    return true;
}

bool leader::wait_ack(int32_t op, ack & a, double timeout_s) {
    (void) a;  // received into by the operation (its receive was posted with it)
    std::string e;
    const auto  t0 = clk::now();
    const bool  ok = p_->wait(op, e, timeout_s);
    st_.back_wait_s += since(t0);
    if (!ok) {
        lose(e);
        err_ = why_;
        return false;
    }
    return true;
}

bool leader::command(ctrl & c, const int32_t * tokens, ack & a, double timeout_s) {
    int32_t op = -1;
    return send_ctrl(c, tokens, op, a) && wait_ack(op, a, timeout_s);
}

run_status leader::set_sampling(int32_t slot, const request_options & o) {
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
    ack a;
    if (!command(c, nullptr, a, ctrl_timeout_s)) {
        return run_status::error;
    }
    if (a.status != int32_t(run_status::ok)) {
        err_ = "the follower: " + why_of(a.why, sizeof a.why);
        return static_cast<run_status>(a.status);
    }
    next_[size_t(slot)].valid = false;
    const run_status rs       = e_->set_sampling(slot, o);
    err_                      = e_->error();
    return rs;
}

run_status leader::prefill(int32_t         slot,
                           const int32_t * tokens,
                           int32_t         n_tokens,
                           int32_t         n_reuse,
                           int32_t &       n_processed) {
    n_processed = 0;
    ctrl c;
    c.op       = int32_t(op::prefill);
    c.slot     = slot;
    c.a        = n_reuse;
    c.n_tokens = n_tokens;
    c.base     = base_;
    base_ += 256;  // a read's chunks never reach 256 (plan_chunks: at most 8,192 rows a chunk)
    ack     a;
    int32_t op = -1;
    if (!send_ctrl(c, tokens, op, a)) {
        return run_status::error;
    }
    x_->expect(slot, c.base);
    next_[size_t(slot)].valid = false;
    // the follower's answer may come first (its read failed): the front stops sending at its next chunk
    bool answered             = false;
    x_->abort                 = [&]() {
        std::string e2;
        if (!answered && !p_->test(op, answered, e2)) {
            answered = true;
        }
        return answered && a.status != int32_t(run_status::ok);
    };
    const run_status rs = e_->prefill(slot, tokens, n_tokens, n_reuse, n_processed);
    x_->abort           = nullptr;
    if (!answered && !wait_ack(op, a, 3600.0)) {
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
            lose(e);
            err_ = why_;
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

int32_t leader::read_begin(int32_t         slot,
                           const int32_t * tokens,
                           int32_t         n_tokens,
                           int32_t         n_reuse,
                           int32_t &       n_processed) {
    n_processed = 0;
    // admission: the follower first (either box's pool or context refuses)
    ctrl c;
    c.op       = int32_t(op::read_begin);
    c.slot     = slot;
    c.a        = n_reuse;
    c.n_tokens = n_tokens;
    c.base     = base_;
    base_ += 256;
    ack a;
    if (!command(c, tokens, a, ctrl_timeout_s)) {
        return -1;
    }
    if (a.status != 0) {
        err_ = "read_begin: the follower: " + why_of(a.why, sizeof a.why);
        return a.status;
    }
    x_->expect(slot, c.base);
    read_base_[size_t(slot)]  = c.base;
    next_[size_t(slot)].valid = false;
    const int32_t r           = e_->read_begin(slot, tokens, n_tokens, n_reuse, n_processed);
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
    return 0;
}

read_status leader::read_step(int32_t slot, int32_t n_groups, read_mode mode) {
    if (lost_) {
        err_ = why_;
        return read_status::error;
    }
    // TTTS1: beside writers or other readers the back's chunks interleave with the step groups there, so the front's
    // end waits for their answers only in the lone mode (else yielded until they came; the drive's tests block)
    const bool block = mode == read_mode::lone || block_reads_;
    if (read_end_[size_t(slot)] >= 0) {
        return read_finish(slot, block);
    }
    const read_status st = e_->read_step(slot, n_groups, mode);
    err_                 = e_->error();
    auto & pend          = read_acks_[size_t(slot)];
    if (st == read_status::chunk_done || st == read_status::done || st == read_status::cancelled) {
        // the front's chunk is done and its crossing sent (or the read stopped): the follower runs the back's half,
        // whole (the lone mode) or interleaved
        ctrl c;
        c.op       = int32_t(op::read_step);
        c.slot     = slot;
        c.a        = mode == read_mode::lone ? 0 : 1;
        c.b        = st == read_status::cancelled ? 1 : 0;
        c.c        = int32_t(x_->sent(slot));  // a cancel: the chunks the front finished run on the back first
        auto *  a  = new ack();
        int32_t op = -1;
        if (!send_ctrl(c, nullptr, op, *a)) {
            delete a;
            return read_status::error;
        }
        pend.push_back({ op, a });
    }
    if (st == read_status::yielded || st == read_status::chunk_done) {
        return st;
    }
    read_end_[size_t(slot)] = int32_t(st);
    return read_finish(slot, block || st == read_status::error);
}

read_status leader::read_finish(int32_t slot, bool block) {
    auto & pend = read_acks_[size_t(slot)];
    if (!block && !lost_) {
        for (auto & pa : pend) {
            if (pa.first < 0) {
                continue;  // answered at an earlier call
            }
            bool        done = false;
            std::string e;
            if (!p_->test(pa.first, done, e)) {
                lose(e);
                break;
            }
            if (!done) {
                return read_status::yielded;
            }
            pa.first = -1 - pa.first;  // answered (the receive finished by test)
        }
    }
    const read_status st    = static_cast<read_status>(read_end_[size_t(slot)]);
    read_end_[size_t(slot)] = -1;
    // the read's end: every back half answered
    read_status out         = lost_ ? read_status::error : st;
    std::string e;
    if (!lost_ && !x_->drain(e)) {
        lose(e);
        out = read_status::error;
    }
    for (auto & [op, a] : pend) {
        if (op >= 0 && (lost_ || !wait_ack(op, *a, 3600.0))) {  // op < 0: answered (read_finish's test)
            out = read_status::error;
        } else if (a->status != int32_t(out == read_status::done ? read_status::chunk_done : out) &&
                   a->status != int32_t(out)) {
            if (out == read_status::done || out == read_status::cancelled) {
                err_ = "read: the follower's back half: " + why_of(a->why, sizeof a->why);
                out  = read_status::error;
            }
        }
        delete a;
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
    e_->read_cancel(slot);  // the front stops at its chunk's end; read_step tells the follower
}

run_status leader::truncate(int32_t slot, int32_t n_keep) {
    // the leader's cut first; the follower's with it (c.b: the leader's slot in place); a slot either box could not
    // keep in place restores at the next prefill on both, so their reads start at the same checkpoint
    next_[size_t(slot)].valid = false;
    const run_status  rs      = e_->truncate(slot, n_keep);
    const std::string lerr    = e_->error();
    ctrl              c;
    c.op   = int32_t(op::truncate);
    c.slot = slot;
    c.a    = n_keep;
    c.b    = rs == run_status::ok && e_->in_place(slot) ? 1 : 0;
    ack a;
    if (!command(c, nullptr, a, ctrl_timeout_s)) {
        return run_status::error;
    }
    if (a.status != int32_t(run_status::ok)) {
        err_ = "truncate: the follower: " + why_of(a.why, sizeof a.why);
        return static_cast<run_status>(a.status);
    }
    if (a.a == 0) {
        e_->to_restore(slot);
    }
    err_ = lerr;
    return rs;
}

run_status leader::release_slot(int32_t slot) {
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

bool leader::follower_pool(pool_info & out) {
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

void leader::bye() {
    if (lost_) {
        p_->close();
        return;
    }
    ctrl c;
    c.op = int32_t(op::bye);
    ack a;
    command(c, nullptr, a, 10.0);
}

std::unique_ptr<leader::pending> leader::group_begin(const int32_t * slots, int32_t n, int32_t group) {
    auto g                   = std::make_unique<pending>();
    g->group                 = group;
    g->n                     = n;
    g->t0                    = clk::now();
    const engine_config & cf = e_->config();
    // each member's checks and cells, and its next step as the last X-STEP left it
    std::vector<int32_t>  lanes;
    for (int32_t k = 0; k < n; ++k) {
        member & m = g->m[k];
        m.slot     = slots[k];
        g->st[k]   = lost_ ? run_status::error : e_->split_enter(slots[k], g->b[k]);
        if (g->st[k] != run_status::ok) {
            g->why[k] = lost_ ? why_ : e_->error();
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
    // the front rounds: the lanes' joint round, then every other member's round alone
    joint_frame * jf    = e_->split_joint();
    bool          joint = false;
    if (lanes.size() >= 2 && jf != nullptr && lanes.size() <= size_t(fn::xchg::max_lanes)) {
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
            fprintf(stderr, "FNFL split: %s; these slots verify alone\n", jerr.c_str());
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
        g->st[k]   = lost_ ? run_status::error : run_status::ok;
        g->why[k]  = lost_ ? why_ : std::string();
        tt.push_back(&e_->test_target(slots[k]));
    }
    if (lost_) {
        for (int32_t k = 0; k < n; ++k) {
            g->m[k].kind = -1;
        }
        return g;
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
    const uint32_t seq = seq_ - 1;  // the command's
    out.assign(size_t(a.a), 0);
    std::string   e;
    const int32_t op = a.a > 0 ? p_->recv(probe_tag(seq), out.data(), out.size(), e) : -1;
    if (a.a > 0 && (op < 0 || !p_->wait(op, e, ctrl_timeout_s))) {
        lose(e);
        err_ = why_;
        return false;
    }
    return true;
}

run_status leader::reconfigure(ctrl c) {
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
    e_ = &e;
    next_.assign(size_t(e.n_slots()), next_step());
    read_acks_.assign(size_t(e.n_slots()), {});
    read_base_.assign(size_t(e.n_slots()), 0);
    read_end_.assign(size_t(e.n_slots()), -1);
    logits_.assign(size_t(e.n_slots()), {});
}

run_status leader::test_prompt(int32_t slot, const std::vector<int32_t> & tokens) {
    target &    t = e_->test_target(slot);
    std::string e;
    if (!t.reset(e)) {
        err_ = "test_prompt: the front's reset: " + e;
        return run_status::error;
    }
    t.set_prompt(tokens);
    next_[size_t(slot)].valid = false;
    ctrl c;
    c.op       = int32_t(op::test_prompt);
    c.slot     = slot;
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

// the group to the follower: its command, the members' boundaries, the answers' receives (sent: all went)
void leader::dispatch(pending & gg, int32_t group) {
    pending *     g = &gg;
    const int32_t n = g->n;
    ctrl &        c = g->c;
    c.op            = int32_t(op::step_group);
    c.group         = group;
    int32_t nm      = 0;
    for (int32_t k = 0; k < n; ++k) {
        if (g->m[k].kind >= 0) {
            c.m[nm++] = g->m[k];
        }
    }
    c.n_members = nm;
    if (nm == 0) {
        return;
    }
    if (lost_) {
        for (int32_t k = 0; k < n; ++k) {
            g->st[k]  = run_status::error;
            g->why[k] = why_;
        }
        return;
    }
    g->seq = seq_++;
    trace("leader", "command " + std::to_string(g->seq) + " step group " + std::to_string(group) + ", " +
                        std::to_string(nm) + " members");
    std::string e;
    bool        ok = true;
    for (int32_t k = 0; k < n && ok; ++k) {
        if (g->m[k].kind < 0) {
            continue;
        }
        g->rops[k] = p_->recv(step_tag(group, g->seq, g->m[k].slot), &g->out[k], sizeof(step_out), e);
        ok         = g->rops[k] >= 0;
        if (ok && g->m[k].kind == int32_t(mkind::forced)) {
            // a forced member's back logits: both cards' rows
            g->lg[k].assign(size_t(2) * size_t(g->m[k].rows) * size_t(fn::head::n_vocab_card), 0.0f);
            g->lops[k] = p_->recv(logits_tag(group, g->seq, g->m[k].slot), g->lg[k].data(), g->lg[k].size() * 4, e);
            ok         = g->lops[k] >= 0;
        }
    }
    const int32_t cop = ok ? p_->send(ctrl_tag(g->seq, 0), &c, sizeof(c), e) : -1;
    ok                = ok && cop >= 0;
    if (ok) {
        g->sops.push_back(cop);
    }
    const auto & buf = e_->config().stage_buf;
    for (int32_t k = 0; k < n && ok; ++k) {
        if (!g->fronted[k]) {
            continue;
        }
        for (int32_t card = 0; card < 2 && ok; ++card) {
            const size_t  bytes = size_t(model::stage_bytes(g->m[k].rows));
            const int32_t op    = p_->send(boundary_tag(group, g->seq, card, g->m[k].slot),
                                           buf[size_t(g->m[k].slot)][size_t(card)], bytes, e);
            ok                  = op >= 0;
            g->sops.push_back(op);
            ++st_.boundaries;
            st_.boundary_bytes += int64_t(bytes);
        }
    }
    if (!ok) {
        lose(e);
    }
    g->sent = ok;
    ++st_.groups;
    st_.members += nm;
}

void leader::group_end(pending & g, joint_step_out * res) {
    std::string e;
    const auto  t0 = clk::now();
    for (int32_t op : g.sops) {
        if (g.sent && !p_->wait(op, e, step_timeout_s)) {
            lose(e);
            g.sent = false;
        }
    }
    for (int32_t k = 0; k < g.n; ++k) {
        res[k]     = joint_step_out();
        member & m = g.m[k];
        if (m.kind < 0 || !g.sent) {
            if (g.fronted[k]) {
                e_->test_target(m.slot).front_abandon(g.fr[k]);
            }
            const std::string why = !g.sent && m.kind >= 0 ? why_ : g.why[k];
            step_info         s;
            res[k].st = g.st[k] == run_status::ok ? run_status::error : g.st[k];
            if (g.st[k] == run_status::ok && m.kind != int32_t(mkind::forced)) {
                e_->split_leave(m.slot, false, s, res[k].out, res[k].n_out, res[k].end);
            }
            res[k].why                  = why;
            next_[size_t(m.slot)].valid = false;
            continue;
        }
        if (!p_->wait(g.rops[k], e, step_timeout_s) || (g.lops[k] >= 0 && !p_->wait(g.lops[k], e, step_timeout_s))) {
            lose(e);
            if (g.fronted[k]) {
                e_->test_target(m.slot).front_abandon(g.fr[k]);
            }
            step_info s;
            if (m.kind != int32_t(mkind::forced)) {
                e_->split_leave(m.slot, false, s, res[k].out, res[k].n_out, res[k].end);
            }
            res[k].st                   = run_status::error;
            res[k].why                  = why_;
            next_[size_t(m.slot)].valid = false;
            continue;
        }
        const step_out & o  = g.out[k];
        target &         t  = e_->test_target(m.slot);
        next_step &      nx = next_[size_t(m.slot)];
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
            logits_[size_t(m.slot)] = std::move(g.lg[k]);
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

void pipeline::end_group(int32_t g) {
    joint_step_out out[max_members];
    l_->group_end(*fl_[g], out);
    for (size_t k = 0; k < members_[g].size(); ++k) {
        forget(members_[g][k]);
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
    const auto kept = [this](int32_t s) {
        for (const auto & k : kept_) {
            if (k.first == s) {
                return true;
            }
        }
        return false;
    };
    bool                 any_kept = false;
    std::vector<int32_t> w;
    for (int32_t k = 0; k < n; ++k) {
        if (kept(slots[k])) {
            any_kept = true;
        } else {
            w.push_back(slots[k]);
        }
    }
    core_.set_writers(w);
    while (!any_kept) {
        const pipe::action x = core_.next();
        if (x.a == pipe::act::begin) {
            fl_[x.g]      = l_->group_begin(x.members.data(), int32_t(x.members.size()), x.g);
            members_[x.g] = x.members;
            core_.done(x);
            continue;
        }
        if (x.a == pipe::act::end) {
            end_group(x.g);
        }
        break;  // an end, or nothing to do (front reads are the shell's own calls)
    }
    for (int32_t k = 0; k < n; ++k) {
        done[k] = false;
        for (auto it = kept_.begin(); it != kept_.end(); ++it) {
            if (it->first == slots[k]) {
                res[k]  = it->second;
                done[k] = true;
                kept_.erase(it);
                break;
            }
        }
    }
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

void pipeline::forget(int32_t slot) {
    kept_.erase(std::remove_if(kept_.begin(), kept_.end(), [slot](const auto & k) { return k.first == slot; }),
                kept_.end());
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
    std::string   e;
    const int32_t op = p_->send(ack_tag(seq), &a, sizeof(a), e);
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
    if (lanes.size() >= 2 && jf != nullptr) {
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
                fprintf(stderr, "FNFL joint: %s; these slots draft alone\n", derr.c_str());
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
            const int32_t op =
                p_->recv(boundary_tag(c.group, seq, card, m.slot), cf.stage_buf[size_t(m.slot)][size_t(card)],
                         size_t(model::stage_bytes(rows)), e);
            ops.push_back(op);
            ok = op >= 0;
        }
    }
    for (int32_t op : ops) {
        ok = ok && p_->wait(op, e, step_timeout_s);
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
        auto lg = std::make_unique<std::vector<float>>(size_t(2) * size_t(rows) * size_t(VC), 0.0f);
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
        outs_.push_back(std::make_unique<step_out>(o));
        out_ops_.push_back(p_->send(step_tag(c.group, seq, c.m[k].slot), outs_.back().get(), sizeof(step_out), e));
        lgs_.push_back(std::move(lg));
        out_ops_.push_back(
            p_->send(logits_tag(c.group, seq, c.m[k].slot), lgs_.back()->data(), lgs_.back()->size() * 4, e));
    }
}

void follower::step_group(const ctrl & c, uint32_t seq) {
    const int32_t n = std::clamp(c.n_members, 0, max_members);
    if (n > 0 && c.m[0].kind == int32_t(mkind::forced)) {
        forced_group(c, seq);
        return;
    }
    const auto             t0 = clk::now();
    const engine_config &  cf = e_->config();
    std::vector<step_out>  out(static_cast<size_t>(n));
    std::vector<step_info> s(static_cast<size_t>(n));
    std::vector<int32_t>   b(size_t(n), 0);
    std::vector<bool>      ran(size_t(n), false), failed(size_t(n), false), malformed(size_t(n), false);
    std::string            e;
    // the members' boundaries
    std::vector<int32_t>   ops;
    for (int32_t k = 0; k < n; ++k) {
        const member & m     = c.m[k];
        out[size_t(k)].slot  = m.slot;
        malformed[size_t(k)] = m.slot < 0 || m.slot >= e_->n_slots() ||
                               (m.kind != int32_t(mkind::begin) && (m.rows < 1 || m.rows > fn::round::max_rows));
        if (m.kind == int32_t(mkind::begin) || malformed[size_t(k)]) {
            continue;
        }
        for (int32_t card = 0; card < 2; ++card) {
            const int32_t op =
                p_->recv(boundary_tag(c.group, seq, card, m.slot), cf.stage_buf[size_t(m.slot)][size_t(card)],
                         size_t(model::stage_bytes(m.rows)), e);
            ops.push_back(op);
        }
    }
    bool have = true;
    for (int32_t op : ops) {
        have = have && op >= 0 && p_->wait(op, e, step_timeout_s);
    }
    trace("follower", "step group " + std::to_string(c.group) + ": boundaries " + (have ? "in" : "failed: " + e));
    const auto fail_member = [&](int32_t k, run_status st, const std::string & why) {
        out[size_t(k)].status = int32_t(st);
        set_why(out[size_t(k)].why, sizeof out[size_t(k)].why, why);
        failed[size_t(k)] = true;
    };
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
                const bool  ok = runtime::step(t, b[size_t(k)], cf.spec_d, cf.eog, s[size_t(k)], why);
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
            si.end                      = is_eog(cf.eog, r.out[0]);
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
        if (!step_cut(t, cf.eog, r, t0, s[size_t(k)], why)) {
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
    if (cm.size() >= 2 && jf != nullptr) {
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
                fprintf(stderr, "FNFL joint: %s; these slots catch up alone\n", cerr.c_str());
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
        step_caught(cf.eog, cr[size_t(k)], t0, s[size_t(k)]);
        ran[size_t(k)] = true;
    }
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
    // the answers (each record lives until its send completes: the next command waits for them)
    for (int32_t k = 0; k < n; ++k) {
        outs_.push_back(std::make_unique<step_out>(out[size_t(k)]));
        const int32_t op = p_->send(step_tag(c.group, seq, c.m[k].slot), outs_.back().get(), sizeof(step_out), e);
        out_ops_.push_back(op);
    }
}

void follower::read_group() {
    iread & r = reads_.front();
    if (r.cancel && !r.cancel_sent) {
        // the chunks the front finished before its cancel took effect (their crossings sent) run here first, then
        // the cancel: both boxes keep the same chunks
        read_info  ri;
        const bool live = e_->get_read_info(r.slot, ri) && ri.live;
        if (live && (x_->taken(r.slot) < r.want || ri.in_chunk)) {
            const read_status st =
                e_->read_step(r.slot, r.lone ? fn::n_layer : 1, r.lone ? read_mode::lone : read_mode::writers);
            if (st == read_status::yielded || st == read_status::chunk_done) {
                return;  // the next turn goes on
            }
            ack a;
            // the read ended on the back with the front's chunks (done: they were all of them): the cancel's end
            a.status = int32_t(st == read_status::done ? read_status::cancelled : st);
            if (st == read_status::error) {
                set_why(a.why, sizeof a.why, e_->error());
            }
            const uint32_t seq = r.seq;
            reads_.pop_front();
            reply(seq, a);
            return;
        }
        e_->read_cancel(r.slot);
        r.cancel_sent = true;
    }
    read_status st = read_status::yielded;
    if (r.lone) {
        while (st == read_status::yielded || (r.cancel && st == read_status::chunk_done)) {
            st = e_->read_step(r.slot, fn::n_layer, read_mode::lone);
        }
    } else {
        st = e_->read_step(r.slot, 1, read_mode::writers);
        if (st == read_status::yielded || (r.cancel && st == read_status::chunk_done)) {
            return;  // the chunk (or the cancel's) goes on at the next turn
        }
    }
    ack a;
    a.status = int32_t(st);
    if (st == read_status::error) {
        set_why(a.why, sizeof a.why, e_->error());
    }
    const uint32_t seq = r.seq;
    reads_.pop_front();
    reply(seq, a);
}

bool follower::serve(std::string & why) {
    uint32_t             seq = 1;
    std::vector<int32_t> tokens;
    for (;;) {
        // the previous answers' sends complete before their records go
        for (int32_t op : out_ops_) {
            std::string e;
            if (op >= 0 && !p_->wait(op, e, step_timeout_s)) {
                why = e;
                return false;
            }
        }
        out_ops_.clear();
        outs_.clear();
        lgs_.clear();
        ctrl        c;
        std::string e;
        bool        got = false;
        int32_t     cop = -1;
        while (!got) {
            if (cop < 0) {
                cop = p_->recv(ctrl_tag(seq, 0), &c, sizeof(c), e);
                if (cop < 0) {
                    why = e;
                    return false;
                }
            }
            if (!reads_.empty()) {
                // read chunks wait: a layer group of one while no command has come
                bool done = false;
                if (!p_->test(cop, done, e)) {
                    why = e;
                    return false;
                }
                if (done) {
                    got = true;
                } else if (p_->lost()) {
                    why = "the leader was lost";
                    return false;
                } else {
                    read_group();
                }
                continue;
            }
            got = p_->wait(cop, e, idle_poll_s);
            cop = -1;  // finished either way (a timed-out wait abandons it)
            if (!got && p_->lost()) {
                why = e;
                return false;
            }
        }
        trace("follower",
              "command " + std::to_string(seq) + " op " + std::to_string(c.op) + " slot " + std::to_string(c.slot));
        if (c.n_tokens > 0) {
            tokens.assign(size_t(c.n_tokens), 0);
            const int32_t op = p_->recv(ctrl_tag(seq, 1), tokens.data(), size_t(c.n_tokens) * 4, e);
            if (op < 0 || !p_->wait(op, e, ctrl_timeout_s)) {
                why = e;
                return false;
            }
        }
        ack        a;
        const auto status = [&](run_status st) {
            a.status = int32_t(st);
            if (st != run_status::ok) {
                set_why(a.why, sizeof a.why, e_->error());
            }
        };
        const op   o = static_cast<op>(c.op);
        // a step group names its slots in its members (checked by split_enter); the others name one, or none
        const bool slotless =
            o == op::pool || o == op::bye || o == op::set_draft || o == op::step_group || o == op::reconfigure;
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
                        discard(c.slot);
                        status(e_->set_sampling(c.slot, ro));
                        break;
                    }
                case op::prefill:
                    {
                        int32_t np = 0;
                        x_->expect(c.slot, c.base);
                        discard(c.slot);
                        status(e_->prefill(c.slot, tokens.data(), c.n_tokens, c.a, np));
                        a.a = np;
                        break;
                    }
                case op::read_begin:
                    {
                        int32_t np = 0;
                        x_->expect(c.slot, c.base);
                        discard(c.slot);
                        a.status = e_->read_begin(c.slot, tokens.data(), c.n_tokens, c.a, np);
                        a.a      = np;
                        if (a.status != 0) {
                            set_why(a.why, sizeof a.why, e_->error());
                        }
                        break;
                    }
                case op::read_step:
                    {
                        // a back chunk (c.a 0: whole, in the lone geometry; 1: interleaved), queued behind the earlier
                        // ones and answered at its end (read_group)
                        iread r;
                        r.slot   = c.slot;
                        r.seq    = seq;
                        r.lone   = c.a == 0;
                        r.cancel = c.b != 0;
                        r.want   = c.c;
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
                    discard(c.slot);
                    status(e_->truncate(c.slot, c.a));
                    if (c.b == 0) {
                        e_->to_restore(c.slot);
                    }
                    a.a = e_->in_place(c.slot) ? 1 : 0;
                    break;
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
                    step_group(c, seq);
                    ++seq;
                    if (!reads_.empty()) {
                        read_group();  // a read layer group after each step group's back half
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
                            out_ops_.push_back(p_->send(probe_tag(seq), probe_out_.data(), probe_out_.size(), e));
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
                case op::test_prompt:
                    {
                        target &    t = e_->test_target(c.slot);
                        std::string te;
                        discard(c.slot);
                        const bool tok = t.reset(te);
                        if (tok) {
                            t.set_prompt(c.n_tokens > 0 ? tokens : std::vector<int32_t>());
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
