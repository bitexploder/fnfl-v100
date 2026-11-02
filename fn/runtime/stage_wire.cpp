#include "fn/runtime/stage_wire.h"

#include "fn/xchg/life.h"

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

double since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

bool fail(std::string & err, const std::string & what) {
    err = what;
    return false;
}

// ---- the link's port ----

class link_port_impl : public port {
  public:
    explicit link_port_impl(lk::link & l) : l_(l) {}

    int32_t send(uint64_t tag, const void * p, size_t n, keep_ref keep, std::string & err) override {
        return l_.send(tag, p, n, err, std::move(keep));
    }

    int32_t recv(uint64_t tag, void * p, size_t n, keep_ref keep, std::string & err) override {
        return l_.recv(tag, p, n, err, std::move(keep));
    }

    bool wait(int32_t op, std::string & err, double timeout_s) override { return l_.wait(op, err, timeout_s); }

    bool test(int32_t op, bool & done, std::string & err) override { return l_.test(op, done, err); }

    int32_t retire(const int32_t * ops, int32_t n) override { return l_.retire(ops, n); }

    bool map(void * p, size_t n, std::string & err) override { return l_.map(p, n, err); }

    bool lost() const override { return l_.lost(); }

    void close() override { l_.quiesce("split stage lost: the leader gave up the stage"); }

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

    // a send completes at once, into the peer's posted receive or parked for it (the bytes copied now): keep is not
    // needed past this call
    int32_t send(uint64_t tag, const void * p, size_t n, keep_ref keep, std::string & err) override {
        (void) keep;
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

    // a receive is written only under the hub's lock, by the peer's send; retire and a timed-out wait unpost it
    int32_t recv(uint64_t tag, void * p, size_t n, keep_ref keep, std::string & err) override {
        (void) keep;
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

    int32_t retire(const int32_t * ops, int32_t n) override {
        std::lock_guard<std::mutex> g(h_->mu);
        for (int32_t k = 0; k < n; ++k) {
            auto it = h_->ops[s_].find(ops[k]);
            if (it == h_->ops[s_].end()) {
                continue;
            }
            auto & ord = h_->order[s_];
            ord.erase(std::remove(ord.begin(), ord.end(), ops[k]), ord.end());
            h_->ops[s_].erase(it);
        }
        return 0;
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

}  // namespace

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

void trace(const char * who, const std::string & what) {
    static const bool on = getenv("FNFL_SPLIT_TRACE") != nullptr;
    if (on) {
        fprintf(stderr, "FNFL split trace: %s: %s\n", who, what.c_str());
        fflush(stderr);
    }
}

std::unique_ptr<port> link_port(lk::link & l) {
    return std::make_unique<link_port_impl>(l);
}

void loop_pair(std::unique_ptr<port> & a, std::unique_ptr<port> & b) {
    auto h = std::make_shared<loop_hub>();
    a      = std::make_unique<loop_port>(h, 0);
    b      = std::make_unique<loop_port>(h, 1);
}

// ---- the wire paths ----

cmd::cmd() {
    life::own(this, sizeof(*this), "command record");
}

cmd::~cmd() {
    if (!tokens.empty()) {
        life::disown(tokens.data(), "command tokens");
    }
    life::disown(this, "command record");
}

group_wire::group_wire() {
    life::own(this, sizeof(*this), "group wire");
}

group_wire::~group_wire() {
    for (const std::vector<float> & v : lg) {
        if (!v.empty()) {
            life::disown(v.data(), "forced logits");
        }
    }
    life::disown(this, "group wire");
}

bool wait_all(port & p, std::vector<int32_t> & ops, double timeout_s, std::string & err) {
    for (int32_t & op : ops) {
        if (op < 0) {
            continue;
        }
        const int32_t o = op;
        op              = -1;                           // finished by the wait either way
        if (!p.wait(o, err, timeout_s)) {
            p.retire(ops.data(), int32_t(ops.size()));  // the rest: none left touching its buffer
            for (int32_t & r : ops) {
                r = -1;
            }
            return false;
        }
    }
    return true;
}

namespace {

// every op in ops not finished retired; each entry -1 after
void retire_all(port & p, int32_t * ops, int32_t n) {
    p.retire(ops, n);
    for (int32_t k = 0; k < n; ++k) {
        ops[k] = -1;
    }
}

}  // namespace

void control::lose(const std::string & why) {
    if (!lost_) {
        lost_ = true;
        why_  = why.rfind("split stage lost", 0) == 0 ? why : "split stage lost: " + why;
        fprintf(stderr, "FNFL split: %s; every later request is refused until the leader restarts\n", why_.c_str());
        fflush(stderr);
        if (p_ != nullptr) {
            p_->close();  // the transport stops touching memory: the other in-flight operations end promptly
        }
    }
}

std::shared_ptr<cmd> control::send(const ctrl & c, const int32_t * tokens) {
    if (lost_) {
        return nullptr;
    }
    // the command's records: one heap object, kept by each of its operations until that operation's end
    auto k = std::make_shared<cmd>();
    k->seq = seq_++;
    k->c   = c;
    if (c.n_tokens > 0) {
        k->tokens.assign(tokens, tokens + c.n_tokens);
        life::own(k->tokens.data(), k->tokens.size() * 4, "command tokens");
    }
    trace("leader",
          "command " + std::to_string(k->seq) + " op " + std::to_string(c.op) + " slot " + std::to_string(c.slot));
    std::string e;
    k->ack_op = p_->recv(ack_tag(k->seq), &k->a, sizeof(k->a), k, e);
    k->c_op   = k->ack_op < 0 ? -1 : p_->send(ctrl_tag(k->seq, 0), &k->c, sizeof(k->c), k, e);
    k->t_op   = k->c_op < 0 || c.n_tokens <= 0 ?
                    -1 :
                    p_->send(ctrl_tag(k->seq, 1), k->tokens.data(), k->tokens.size() * 4, k, e);
    bool ok   = k->ack_op >= 0 && k->c_op >= 0 && (c.n_tokens <= 0 || k->t_op >= 0);
    for (int32_t * op : { &k->c_op, &k->t_op }) {
        if (ok && *op >= 0) {
            const int32_t o = *op;
            *op             = -1;
            ok              = p_->wait(o, e, send_timeout_s);
        }
    }
    if (!ok) {
        int32_t all[3] = { k->ack_op, k->c_op, k->t_op };
        retire_all(*p_, all, 3);
        k->ack_op = k->c_op = k->t_op = -1;
        lose(e);
        return nullptr;
    }
    return k;
}

bool control::wait(cmd & k, double timeout_s) {
    if (k.answered) {
        return true;
    }
    if (k.ack_op < 0) {
        return false;  // dropped
    }
    std::string   e;
    const auto    t0 = clk::now();
    const int32_t op = k.ack_op;
    k.ack_op         = -1;  // finished by the wait either way (a give-up retired it)
    const bool ok    = p_->wait(op, e, timeout_s);
    wait_s += since(t0);
    if (!ok) {
        lose(e);
        return false;
    }
    k.answered = true;
    return true;
}

bool control::test(cmd & k, bool & done) {
    done = k.answered;
    if (done) {
        return true;
    }
    if (k.ack_op < 0) {
        done = true;
        return false;
    }
    std::string e;
    if (!p_->test(k.ack_op, done, e)) {
        k.ack_op = -1;
        lose(e);
        done = true;
        return false;
    }
    if (done) {
        k.ack_op   = -1;
        k.answered = true;
    }
    return true;
}

void control::drop(cmd & k) {
    if (k.ack_op >= 0) {
        p_->retire(&k.ack_op, 1);
        k.ack_op = -1;
    }
}

bool control::command(ctrl & c, const int32_t * tokens, ack & a, double timeout_s) {
    const std::shared_ptr<cmd> k = send(c, tokens);
    if (k == nullptr || !wait(*k, timeout_s)) {
        return false;
    }
    a = k->a;  // the answer copied out: the caller's record never meets the transport
    return true;
}

bool post_group(control &                           ctl,
                const std::shared_ptr<group_wire> & wp,
                int32_t                             group,
                const member *                      m,
                int32_t                             n,
                const boundary_src *                src,
                const size_t *                      logits_n,
                leader_stats &                      st) {
    group_wire & g = *wp;
    port &       p = *ctl.p();
    ctrl &       c = g.c;
    c.op           = int32_t(op::step_group);
    c.group        = group;
    int32_t nm     = 0;
    for (int32_t k = 0; k < n; ++k) {
        if (m[k].kind >= 0) {
            c.m[nm++] = m[k];
        }
    }
    c.n_members = nm;
    g.sent      = false;
    if (nm == 0 || ctl.lost()) {
        return false;
    }
    g.seq = ctl.take_seq();
    trace("leader", "command " + std::to_string(g.seq) + " step group " + std::to_string(group) + ", " +
                        std::to_string(nm) + " members");
    // every operation keeps the group's records (wp); the boundary rows are the slots' pinned buffers, refilled only
    // at the slot's next front round, after this group's end
    std::string e;
    bool        ok = true;
    for (int32_t k = 0; k < n && ok; ++k) {
        if (m[k].kind < 0) {
            continue;
        }
        g.rops[k] = p.recv(step_tag(group, g.seq, m[k].slot), &g.out[k], sizeof(step_out), wp, e);
        ok        = g.rops[k] >= 0;
        if (ok && logits_n != nullptr && logits_n[k] > 0) {
            g.lg[k].assign(logits_n[k], 0.0f);
            life::own(g.lg[k].data(), g.lg[k].size() * 4, "forced logits");
            g.lops[k] = p.recv(logits_tag(group, g.seq, m[k].slot), g.lg[k].data(), g.lg[k].size() * 4, wp, e);
            ok        = g.lops[k] >= 0;
        }
    }
    const int32_t cop = ok ? p.send(ctrl_tag(g.seq, 0), &c, sizeof(c), wp, e) : -1;
    ok                = ok && cop >= 0;
    if (ok) {
        g.sops.push_back(cop);
    }
    for (int32_t k = 0; k < n && ok; ++k) {
        if (src == nullptr || src[k].bytes == 0) {
            continue;
        }
        for (int32_t card = 0; card < 2 && ok; ++card) {
            const int32_t op =
                p.send(boundary_tag(group, g.seq, card, m[k].slot), src[k].buf[card], src[k].bytes, wp, e);
            ok = op >= 0;
            if (ok) {
                g.sops.push_back(op);
                ++st.boundaries;
                st.boundary_bytes += int64_t(src[k].bytes);
            }
        }
    }
    if (!ok) {
        // a submission refused: everything this group submitted ends before the call returns
        group_close(ctl, g);
        ctl.lose(e);
    }
    g.sent = ok;
    ++st.groups;
    st.members += nm;
    return ok;
}

bool group_sends(control & ctl, group_wire & g, double timeout_s) {
    std::string e;
    for (int32_t & op : g.sops) {
        if (!g.sent || op < 0) {
            continue;
        }
        const int32_t o = op;
        op              = -1;
        if (!ctl.p()->wait(o, e, timeout_s)) {
            g.sent = false;
            group_close(ctl, g);  // the answers' receives too: nothing of the group touches its records after
            ctl.lose(e);
        }
    }
    return g.sent;
}

bool group_answer(control & ctl, group_wire & g, int32_t k, double timeout_s, std::string & e) {
    port & p = *ctl.p();
    for (int32_t * op : { &g.rops[k], &g.lops[k] }) {
        if (*op < 0) {
            continue;
        }
        const int32_t o = *op;
        *op             = -1;
        if (!p.wait(o, e, timeout_s)) {
            group_close(ctl, g);
            ctl.lose(e);
            return false;
        }
    }
    return true;
}

void group_close(control & ctl, group_wire & g) {
    port & p = *ctl.p();
    retire_all(p, g.rops, max_members);
    retire_all(p, g.lops, max_members);
    if (!g.sops.empty()) {
        retire_all(p, g.sops.data(), int32_t(g.sops.size()));
    }
}

}  // namespace fn::runtime::stage
