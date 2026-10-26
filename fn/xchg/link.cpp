#include "fn/xchg/link.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <ucp/api/ucp.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

namespace fn::xchg::link {

namespace {

using clk = std::chrono::steady_clock;

constexpr int32_t   n_ops     = 256;  // operation slots (in flight at once)
constexpr uint32_t  q_cap     = 512;  // submission queue entries (a power of two)
constexpr ucp_tag_t full_mask = ~ucp_tag_t(0);

double since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

bool fail(std::string & err, const std::string & what) {
    err = what;
    return false;
}

void relax() {
#if defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}

// an operation's states: the runtime allocates a free slot and queues it; the link thread posts it and publishes its
// completion (release); the runtime acquires that and frees the slot. An abandoned op (a wait that timed out) becomes
// reclaimable only once the link thread has seen its request end, so its buffer is never written after reuse
enum op_state : int32_t { st_free = 0, st_queued = 1, st_posted = 2, st_ok = 3, st_err = 4, st_reclaim = 5 };

// submission entries: kind in the top 4 bits, the op's generation (20 bits) and its index (8 bits) below, so an entry
// queued for an earlier use of a slot is recognized and skipped
enum sub_kind : uint32_t { sub_post = 1, sub_cancel = 2, sub_job = 3, sub_stop = 4 };

static_assert(n_ops <= 256, "an entry carries the op index in 8 bits");

uint32_t sub(sub_kind k, uint32_t i, uint32_t gen = 0) {
    return (uint32_t(k) << 28) | ((gen & 0xfffffu) << 8) | (i & 0xffu);
}

}  // namespace

const char * kind_name(kind k) {
    switch (k) {
        case kind::hello:
            return "X-HELLO";
        case kind::boundary:
            return "X-BOUNDARY";
        case kind::prompt:
            return "X-BOUNDARY(prompt)";
        case kind::step:
            return "X-STEP";
        case kind::ctrl:
            return "X-CTRL";
        case kind::ack:
            return "X-ACK";
        case kind::bye:
            return "X-BYE";
        case kind::test:
            return "X-TEST";
    }
    return "X-?";
}

uint64_t make_tag(kind k, uint8_t group, uint32_t seq, uint8_t card, uint8_t aux) {
    return uint64_t(uint8_t(k)) << 56 | uint64_t(group) << 48 | uint64_t(seq) << 16 | uint64_t(card) << 8 | aux;
}

tag_fields split_tag(uint64_t t) {
    tag_fields f;
    f.k     = static_cast<kind>(uint8_t(t >> 56));
    f.group = uint8_t(t >> 48);
    f.seq   = uint32_t(t >> 16);
    f.card  = uint8_t(t >> 8);
    f.aux   = uint8_t(t);
    return f;
}

std::string tag_str(uint64_t t) {
    const tag_fields f = split_tag(t);
    char             b[128];
    std::snprintf(b, sizeof b, "%s group %u seq %u card %u aux %u", kind_name(f.k), unsigned(f.group), unsigned(f.seq),
                  unsigned(f.card), unsigned(f.aux));
    return b;
}

void set_text(char * dst, size_t cap, const std::string & s) {
    std::memset(dst, 0, cap);
    std::memcpy(dst, s.data(), std::min(cap - 1, s.size()));
}

std::string hello_mismatch(const hello & a, const hello & b) {
    const auto txt = [](const char * p, size_t cap) {
        return std::string(p, strnlen(p, cap));
    };
    if (a.magic != b.magic || a.version != b.version) {
        return "magic or version (" + std::to_string(a.magic) + "/" + std::to_string(a.version) + " vs " +
               std::to_string(b.magic) + "/" + std::to_string(b.version) + ")";
    }
    if (std::memcmp(a.commit, b.commit, sizeof a.commit) != 0) {
        return "source commit (" + txt(a.commit, sizeof a.commit) + " vs " + txt(b.commit, sizeof b.commit) + ")";
    }
    if (std::memcmp(a.digest, b.digest, sizeof a.digest) != 0) {
        return "device-code digest (" + txt(a.digest, sizeof a.digest) + " vs " + txt(b.digest, sizeof b.digest) + ")";
    }
    if (std::memcmp(a.model, b.model, sizeof a.model) != 0) {
        return "model header hash (" + txt(a.model, sizeof a.model) + " vs " + txt(b.model, sizeof b.model) + ")";
    }
    if (a.split != b.split) {
        return "split layer k (" + std::to_string(a.split) + " vs " + std::to_string(b.split) + ")";
    }
    if (a.slots != b.slots) {
        return "slot count (" + std::to_string(a.slots) + " vs " + std::to_string(b.slots) + ")";
    }
    if (a.role == b.role) {
        return "role (both " + std::string(a.role == 0 ? "leader" : "follower") + ")";
    }
    return std::string();
}

struct link::impl {
    struct op {
        std::atomic<int32_t> state{ st_free };
        bool                 is_send = false;
        uint64_t             tag     = 0;
        void *               p       = nullptr;
        size_t               n       = 0;
        // written by the link thread before the state's release store
        ucs_status_t         status  = UCS_OK;
        size_t               got     = 0;
        uint64_t             got_tag = 0;
        void *               req     = nullptr;
        bool     abandoned = false;  // set by the runtime before it queues a cancel (the link thread reads it after)
        uint32_t gen       = 0;      // the slot's use count (the queue entries carry it)
        ucp_tag_recv_info_t info = {};
    };

    options            o;
    hello              theirs;
    ucp_context_h      ctx      = nullptr;
    ucp_worker_h       worker   = nullptr;
    ucp_listener_h     listener = nullptr;
    ucp_ep_h           ep       = nullptr;
    ucp_conn_request_h conn     = nullptr;
    int                efd = -1, evfd = -1, epfd = -1;
    std::thread        th;
    std::atomic<bool>  running{ false };
    std::atomic<bool>  stopping{ false };
    std::atomic<bool>  sleeping{ false };
    std::atomic<bool>  lost_flag{ false };
    std::mutex         why_mu;
    std::string        why;
    std::string        transport;

    op                    ops[n_ops];
    // the submission queue (single producer: the runtime thread; single consumer: the link thread)
    std::atomic<uint32_t> q_head{ 0 }, q_tail{ 0 };
    uint32_t              q[q_cap] = {};
    // a job the link thread runs for the runtime (map, probe), one at a time
    std::function<void()> job;
    std::atomic<bool>     job_done{ false };

    struct mapped {
        uint8_t * p;
        size_t    n;
        ucp_mem_h h;
    };

    std::vector<mapped> maps;             // link thread only once running (map runs as a job)
    int32_t             outstanding = 0;  // link thread only
    counters            tot;

    // ---- the link thread's side ----
    static void ep_error(void * arg, ucp_ep_h e, ucs_status_t st) {
        impl * me = static_cast<impl *>(arg);
        if (e != me->ep) {
            return;  // a closed connect attempt's endpoint
        }
        {
            std::lock_guard<std::mutex> g(me->why_mu);
            if (me->why.empty()) {
                me->why = std::string("the peer is lost (") + ucs_status_string(st) + ")";
            }
        }
        me->lost_flag.store(true, std::memory_order_release);
    }

    static void on_conn(ucp_conn_request_h r, void * arg) {
        impl * me = static_cast<impl *>(arg);
        if (me->conn == nullptr && me->ep == nullptr) {
            me->conn = r;
        } else {
            ucp_listener_reject(me->listener, r);  // one leader only
        }
    }

    void complete(op & x, ucs_status_t st) {
        x.status = st;
        bool ok  = st == UCS_OK;
        if (ok && !x.is_send) {
            x.got     = x.info.length;
            x.got_tag = x.info.sender_tag;
            ok        = x.got == x.n;
        }
        x.req = nullptr;
        x.state.store(x.abandoned ? st_reclaim : (ok ? st_ok : st_err), std::memory_order_release);
    }

    static void send_cb(void * req, ucs_status_t st, void * user) {
        op & x   = *static_cast<op *>(user);
        x.status = st;
        x.req    = nullptr;
        x.state.store(x.abandoned ? st_reclaim : (st == UCS_OK ? st_ok : st_err), std::memory_order_release);
        ucp_request_free(req);
    }

    static void recv_cb(void * req, ucs_status_t st, const ucp_tag_recv_info_t * info, void * user) {
        op & x = *static_cast<op *>(user);
        if (info != nullptr) {
            x.info = *info;
        }
        bool ok = st == UCS_OK;
        if (ok) {
            x.got     = x.info.length;
            x.got_tag = x.info.sender_tag;
            ok        = x.got == x.n;
        }
        x.status = st;
        x.req    = nullptr;
        x.state.store(x.abandoned ? st_reclaim : (ok ? st_ok : st_err), std::memory_order_release);
        ucp_request_free(req);
    }

    ucp_mem_h memh_of(const void * p, size_t n) const {
        const uint8_t * b = static_cast<const uint8_t *>(p);
        for (const mapped & m : maps) {
            if (b >= m.p && b + n <= m.p + m.n) {
                return m.h;
            }
        }
        return nullptr;
    }

    void post(int32_t i) {
        op &                x   = ops[i];
        ucp_request_param_t prm = {};
        prm.op_attr_mask        = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
        prm.user_data           = &x;
        if (ucp_mem_h h = memh_of(x.p, x.n)) {
            prm.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
            prm.memh = h;
        }
        ucs_status_ptr_t r      = nullptr;
        int32_t          expect = st_queued;
        if (!x.state.compare_exchange_strong(expect, st_posted, std::memory_order_acq_rel)) {
            return;  // abandoned before it was posted (reclaimed by the runtime)
        }
        if (lost_flag.load(std::memory_order_acquire) || ep == nullptr) {
            complete(x, UCS_ERR_CONNECTION_RESET);
            return;
        }
        if (x.is_send) {
            prm.cb.send = send_cb;
            r           = ucp_tag_send_nbx(ep, x.p, x.n, x.tag, &prm);
        } else {
            prm.cb.recv = recv_cb;
            prm.op_attr_mask |= UCP_OP_ATTR_FIELD_RECV_INFO;
            prm.recv_info.tag_info = &x.info;
            r                      = ucp_tag_recv_nbx(worker, x.p, x.n, x.tag, full_mask, &prm);
        }
        if (r == nullptr) {
            complete(x, UCS_OK);
        } else if (UCS_PTR_IS_ERR(r)) {
            complete(x, UCS_PTR_STATUS(r));
        } else {
            x.req = r;
        }
    }

    bool any_posted() const {
        for (const op & x : ops) {
            if (x.state.load(std::memory_order_relaxed) == st_posted) {
                return true;
            }
        }
        return false;
    }

    void run() {
        epoll_event ev[2];
        while (!stopping.load(std::memory_order_acquire)) {
            uint32_t h = q_head.load(std::memory_order_acquire);
            uint32_t t = q_tail.load(std::memory_order_relaxed);
            while (t != h) {
                const uint32_t e = q[t & (q_cap - 1)];
                t++;
                q_tail.store(t, std::memory_order_release);
                const uint32_t k = e >> 28, i = e & 0xffu, gen = (e >> 8) & 0xfffffu;
                if (k == sub_post) {
                    if ((ops[i].gen & 0xfffffu) == gen) {
                        post(int32_t(i));
                    }
                } else if (k == sub_cancel) {
                    op & x = ops[i];
                    if ((x.gen & 0xfffffu) != gen) {
                        // a stale entry: the slot was reused
                    } else if (x.req != nullptr) {
                        ucp_request_cancel(worker, x.req);
                    } else if (x.state.load(std::memory_order_relaxed) != st_posted) {
                        x.state.store(st_reclaim, std::memory_order_release);
                    }
                } else if (k == sub_job) {
                    job();
                    job_done.store(true, std::memory_order_release);
                } else if (k == sub_stop) {
                    stopping.store(true, std::memory_order_release);
                }
                h = q_head.load(std::memory_order_acquire);
            }
            int progressed = 0;
            while (ucp_worker_progress(worker) != 0) {
                ++progressed;
            }
            if (progressed > 0 || any_posted()) {
                relax();
                continue;
            }
            // idle: arm the worker, publish that this thread sleeps, recheck the queue, then sleep
            const ucs_status_t st = ucp_worker_arm(worker);
            if (st == UCS_ERR_BUSY) {
                continue;
            }
            sleeping.store(true, std::memory_order_seq_cst);
            if (q_head.load(std::memory_order_seq_cst) != q_tail.load(std::memory_order_relaxed) ||
                stopping.load(std::memory_order_acquire)) {
                sleeping.store(false, std::memory_order_relaxed);
                continue;
            }
            const int n = epoll_wait(epfd, ev, 2, 100);
            sleeping.store(false, std::memory_order_relaxed);
            for (int j = 0; j < n; ++j) {
                if (ev[j].data.fd == evfd) {
                    uint64_t v  = 0;
                    ssize_t  rd = read(evfd, &v, sizeof v);
                    (void) rd;
                }
            }
        }
        // stopping: flush and close the endpoint, bounded
        if (ep != nullptr) {
            ucp_request_param_t prm = {};
            prm.op_attr_mask        = UCP_OP_ATTR_FIELD_FLAGS;
            prm.flags               = lost_flag.load() ? UCP_EP_CLOSE_FLAG_FORCE : 0;
            ucs_status_ptr_t r      = ucp_ep_close_nbx(ep, &prm);
            const auto       t0     = clk::now();
            if (r != nullptr && UCS_PTR_IS_PTR(r)) {
                while (ucp_request_check_status(r) == UCS_INPROGRESS && since(t0) < 2.0) {
                    ucp_worker_progress(worker);
                }
                ucp_request_free(r);
            }
            ep = nullptr;
        }
    }

    // ---- the runtime's side ----
    bool push(uint32_t e) {
        const uint32_t h = q_head.load(std::memory_order_relaxed);
        if (h - q_tail.load(std::memory_order_acquire) >= q_cap) {
            return false;
        }
        q[h & (q_cap - 1)] = e;
        q_head.store(h + 1, std::memory_order_seq_cst);
        if (sleeping.load(std::memory_order_seq_cst)) {
            const uint64_t one = 1;
            ssize_t        w   = write(evfd, &one, sizeof one);
            (void) w;
        }
        return true;
    }

    void run_job(std::function<void()> f) {
        if (!running.load()) {
            f();  // before the thread starts, the caller owns the worker
            return;
        }
        job = std::move(f);
        job_done.store(false, std::memory_order_release);
        while (!push(sub(sub_job, 0))) {
            relax();
        }
        while (!job_done.load(std::memory_order_acquire)) {
            relax();
        }
    }

    int32_t alloc(std::string & err) {
        for (int32_t i = 0; i < n_ops; ++i) {
            int32_t s = ops[i].state.load(std::memory_order_acquire);
            if (s == st_free || s == st_reclaim) {
                return i;
            }
        }
        err = "link: no free operation slot (" + std::to_string(n_ops) + " in flight)";
        return -1;
    }

    int32_t submit(bool is_send, uint64_t tag, void * p, size_t n, std::string & err) {
        if (lost_flag.load(std::memory_order_acquire)) {
            err = "split stage lost: " + tag_str(tag) + ": " + lost_why();
            return -1;
        }
        const int32_t i = alloc(err);
        if (i < 0) {
            return -1;
        }
        op & x      = ops[i];
        x.is_send   = is_send;
        x.tag       = tag;
        x.p         = p;
        x.n         = n;
        x.status    = UCS_OK;
        x.got       = 0;
        x.got_tag   = 0;
        x.req       = nullptr;
        x.abandoned = false;
        x.info      = {};
        ++x.gen;
        x.state.store(st_queued, std::memory_order_release);
        if (!push(sub(sub_post, uint32_t(i), x.gen))) {
            x.state.store(st_free, std::memory_order_relaxed);
            err = "link: the submission queue is full";
            return -1;
        }
        return i;
    }

    std::string lost_why() {
        std::lock_guard<std::mutex> g(why_mu);
        return why.empty() ? std::string("the peer is lost") : why;
    }

    // the outcome of a completed op (state ok or err), then the slot freed
    bool finish(int32_t i, int32_t s, std::string & err) {
        op & x  = ops[i];
        bool ok = s == st_ok;
        if (!ok) {
            if (x.status == UCS_OK && !x.is_send) {
                err = "refused: " + tag_str(x.tag) + ": received " + std::to_string(x.got) + " bytes, expected " +
                      std::to_string(x.n);
            } else if (x.status == UCS_ERR_MESSAGE_TRUNCATED) {
                err = "refused: " + tag_str(x.tag) + ": the message is longer than the " + std::to_string(x.n) +
                      " bytes expected (truncated)";
            } else if (lost_flag.load(std::memory_order_acquire)) {
                err = "split stage lost: " + tag_str(x.tag) + ": " + lost_why();
            } else {
                err =
                    std::string(x.is_send ? "send " : "receive ") + tag_str(x.tag) + ": " + ucs_status_string(x.status);
            }
        } else if (x.is_send) {
            ++tot.sent_msgs;
            tot.sent_bytes += x.n;
        } else {
            ++tot.recv_msgs;
            tot.recv_bytes += x.got;
        }
        x.state.store(st_free, std::memory_order_release);
        return ok;
    }

    void abandon(int32_t i) {
        ops[i].abandoned = true;
        std::atomic_thread_fence(std::memory_order_release);
        // the cancel entry is ordered after abandoned by the queue's release; the link thread reclaims the slot
        int32_t expect = st_queued;
        if (ops[i].state.compare_exchange_strong(expect, st_reclaim)) {
            return;  // never posted: nothing to cancel (the post entry finds it reclaimed and skips it)
        }
        while (!push(sub(sub_cancel, uint32_t(i), ops[i].gen))) {
            relax();
        }
    }
};

link::link() : p_(std::make_unique<impl>()) {}

link::~link() {
    stop();
}

const hello & link::peer() const {
    return p_->theirs;
}

const std::string & link::transport() const {
    return p_->transport;
}

bool link::lost() const {
    return p_->lost_flag.load(std::memory_order_acquire);
}

std::string link::lost_why() const {
    return p_->lost_why();
}

counters link::totals() const {
    return p_->tot;
}

bool link::start(const options & o, std::string & err) {
    impl & m           = *p_;
    m.o                = o;
    m.o.me.role        = o.leader ? 0 : 1;
    ucp_config_t * cfg = nullptr;
    if (ucp_config_read(nullptr, nullptr, &cfg) != UCS_OK) {
        return fail(err, "link: ucp_config_read");
    }
    ucp_params_t up        = {};
    up.field_mask          = UCP_PARAM_FIELD_FEATURES;
    up.features            = UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP;
    const ucs_status_t ist = ucp_init(&up, cfg, &m.ctx);
    ucp_config_release(cfg);
    if (ist != UCS_OK) {
        return fail(err, std::string("link: ucp_init: ") + ucs_status_string(ist));
    }
    // the worker is UCS_THREAD_MODE_SINGLE: the link thread creates it, connects and exchanges X-HELLO, then runs
    // its loop; the caller waits for the setup's outcome
    const auto setup = [&]() -> bool {
        ucp_worker_params_t wp = {};
        wp.field_mask          = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        wp.thread_mode         = UCS_THREAD_MODE_SINGLE;
        if (ucp_worker_create(m.ctx, &wp, &m.worker) != UCS_OK || ucp_worker_get_efd(m.worker, &m.efd) != UCS_OK) {
            return fail(err, "link: ucp_worker_create or its event descriptor");
        }
        m.evfd = eventfd(0, EFD_NONBLOCK);
        m.epfd = epoll_create1(0);
        if (m.evfd < 0 || m.epfd < 0) {
            return fail(err, "link: eventfd or epoll");
        }
        epoll_event e1 = {};
        e1.events      = EPOLLIN;
        e1.data.fd     = m.efd;
        epoll_event e2 = {};
        e2.events      = EPOLLIN;
        e2.data.fd     = m.evfd;
        if (epoll_ctl(m.epfd, EPOLL_CTL_ADD, m.efd, &e1) != 0 || epoll_ctl(m.epfd, EPOLL_CTL_ADD, m.evfd, &e2) != 0) {
            return fail(err, "link: epoll_ctl");
        }
        sockaddr_in sa = {};
        sa.sin_family  = AF_INET;
        sa.sin_port    = htons(o.port);
        if (inet_pton(AF_INET, o.addr.c_str(), &sa.sin_addr) != 1) {
            return fail(err, "link: bad address " + o.addr);
        }
        const auto ep_err = [&](ucp_ep_params_t & ep) {
            ep.field_mask |= UCP_EP_PARAM_FIELD_ERR_HANDLER | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
            ep.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
            ep.err_handler.cb  = impl::ep_error;
            ep.err_handler.arg = &m;
        };
        const auto t0       = clk::now();
        // the hello exchange on this thread (the link thread does not exist yet, so this thread owns the worker)
        const auto exchange = [&](std::string & e) -> bool {
            hello      mine = m.o.me;
            hello      got;
            impl::op & rs  = m.ops[0];
            impl::op & ss  = m.ops[1];
            const auto arm = [&](impl::op & x, bool snd, void * p) {
                x.is_send   = snd;
                x.tag       = make_tag(kind::hello, 0, 0, 0);
                x.p         = p;
                x.n         = sizeof(hello);
                x.abandoned = false;
                x.info      = {};
                x.state.store(st_queued);
                m.post(snd ? 1 : 0);
            };
            arm(rs, false, &got);
            if (o.leader) {
                arm(ss, true, &mine);
            }
            bool sent = o.leader;  // the leader sent its X-HELLO above; the follower answers once it has the leader's
            const auto tw = clk::now();
            while (since(tw) < std::max(2.0, o.timeout_s)) {
                ucp_worker_progress(m.worker);
                if (m.lost_flag.load()) {
                    break;
                }
                if (!o.leader && rs.state.load() == st_ok && !sent) {
                    arm(ss, true, &mine);  // the follower answers after it has read the leader's
                    sent = true;
                }
                const int32_t a = rs.state.load(), b = ss.state.load();
                if ((a == st_ok || a == st_err) && sent && (b == st_ok || b == st_err)) {
                    break;
                }
            }
            const bool ok = rs.state.load() == st_ok && ss.state.load() == st_ok;
            if (!ok) {
                if (rs.req != nullptr) {
                    ucp_request_cancel(m.worker, rs.req);
                    for (int i = 0; i < 1000 && rs.req != nullptr; ++i) {
                        ucp_worker_progress(m.worker);
                    }
                }
                rs.state.store(st_free);
                ss.state.store(st_free);
                return fail(e, m.lost_flag.load() ? m.lost_why() : "no X-HELLO answer");
            }
            rs.state.store(st_free);
            ss.state.store(st_free);
            m.theirs             = got;
            const std::string mm = hello_mismatch(mine, got);
            if (!mm.empty()) {
                return fail(e, "refused to start: the boxes differ in their " + mm);
            }
            return true;
        };
        if (!o.leader) {
            ucp_listener_params_t lp = {};
            lp.field_mask            = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
            lp.sockaddr.addr         = reinterpret_cast<const sockaddr *>(&sa);
            lp.sockaddr.addrlen      = sizeof sa;
            lp.conn_handler.cb       = impl::on_conn;
            lp.conn_handler.arg      = &m;
            const ucs_status_t ls    = ucp_listener_create(m.worker, &lp, &m.listener);
            if (ls != UCS_OK) {
                return fail(err,
                            "link: listen on " + o.addr + ":" + std::to_string(o.port) + ": " + ucs_status_string(ls));
            }
            while (m.conn == nullptr && since(t0) < o.connect_s) {
                ucp_worker_progress(m.worker);
            }
            if (m.conn == nullptr) {
                return fail(err, "link: no leader connected in " + std::to_string(o.connect_s) + " s");
            }
            ucp_ep_params_t ep = {};
            ep.field_mask      = UCP_EP_PARAM_FIELD_CONN_REQUEST;
            ep.conn_request    = m.conn;
            ep_err(ep);
            if (ucp_ep_create(m.worker, &ep, &m.ep) != UCS_OK) {
                return fail(err, "link: accept the leader");
            }
            if (!exchange(err)) {
                return false;
            }
        } else {
            std::string last  = "no attempt";
            bool        up_ok = false;
            while (since(t0) < o.connect_s) {
                m.lost_flag.store(false);
                {
                    std::lock_guard<std::mutex> g(m.why_mu);
                    m.why.clear();
                }
                ucp_ep_params_t ep  = {};
                ep.field_mask       = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR;
                ep.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
                ep.sockaddr.addr    = reinterpret_cast<const sockaddr *>(&sa);
                ep.sockaddr.addrlen = sizeof sa;
                ep_err(ep);
                if (ucp_ep_create(m.worker, &ep, &m.ep) != UCS_OK) {
                    last = "ucp_ep_create";
                } else if (exchange(last)) {
                    up_ok = true;
                    break;
                } else if (last.rfind("refused to start", 0) == 0) {
                    return fail(err, last);
                }
                if (m.ep != nullptr) {
                    ucp_request_param_t prm = {};
                    prm.op_attr_mask        = UCP_OP_ATTR_FIELD_FLAGS;
                    prm.flags               = UCP_EP_CLOSE_FLAG_FORCE;
                    ucs_status_ptr_t r      = ucp_ep_close_nbx(m.ep, &prm);
                    if (r != nullptr && UCS_PTR_IS_PTR(r)) {
                        while (ucp_request_check_status(r) == UCS_INPROGRESS) {
                            ucp_worker_progress(m.worker);
                        }
                        ucp_request_free(r);
                    }
                    m.ep = nullptr;
                }
                const auto tz = clk::now();
                while (since(tz) < 0.2) {
                    ucp_worker_progress(m.worker);
                }
            }
            if (!up_ok) {
                return fail(err, "link: connect to " + o.addr + ":" + std::to_string(o.port) + " failed in " +
                                     std::to_string(o.connect_s) + " s: " + last);
            }
        }
        // the endpoint's transports
        {
            ucp_transport_entry_t ent[16] = {};
            ucp_ep_attr_t         at      = {};
            at.field_mask                 = UCP_EP_ATTR_FIELD_TRANSPORTS;
            at.transports.entries         = ent;
            at.transports.num_entries     = 16;
            at.transports.entry_size      = sizeof(ucp_transport_entry_t);
            std::string t;
            if (ucp_ep_query(m.ep, &at) == UCS_OK) {
                for (unsigned i = 0; i < at.transports.num_entries; ++i) {
                    const std::string one = std::string(ent[i].transport_name ? ent[i].transport_name : "?") + "/" +
                                            (ent[i].device_name ? ent[i].device_name : "?");
                    if (t.find(one) == std::string::npos) {
                        t += (t.empty() ? "" : " ") + one;
                    }
                }
            }
            m.transport = t.empty() ? std::string("unknown") : t;
        }
        return true;
    };
    std::promise<bool> ready;
    std::future<bool>  ok = ready.get_future();
    m.th                  = std::thread([&m, &setup, &ready]() {
        const bool up = setup();
        if (up) {
            m.running.store(true);
        }
        ready.set_value(up);
        if (up) {
            m.run();
        }
    });
    if (!ok.get()) {
        m.th.join();
        return false;
    }
    return true;
}

bool link::map(void * p, size_t n, std::string & err) {
    impl &       m  = *p_;
    ucs_status_t st = UCS_OK;
    ucp_mem_h    h  = nullptr;
    m.run_job([&]() {
        ucp_mem_map_params_t mp = {};
        mp.field_mask           = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        mp.address              = p;
        mp.length               = n;
        st                      = ucp_mem_map(m.ctx, &mp, &h);
        if (st == UCS_OK) {
            m.maps.push_back({ static_cast<uint8_t *>(p), n, h });
        }
    });
    if (st != UCS_OK) {
        return fail(err, std::string("link: ucp_mem_map of ") + std::to_string(n) + " bytes: " + ucs_status_string(st));
    }
    return true;
}

int32_t link::send(uint64_t tag, const void * p, size_t n, std::string & err) {
    return p_->submit(true, tag, const_cast<void *>(p), n, err);
}

int32_t link::recv(uint64_t tag, void * p, size_t n, std::string & err) {
    return p_->submit(false, tag, p, n, err);
}

bool link::test(int32_t op, bool & done, std::string & err) {
    impl & m = *p_;
    done     = false;
    if (op < 0 || op >= n_ops) {
        return fail(err, "link: bad operation " + std::to_string(op));
    }
    const int32_t s = m.ops[op].state.load(std::memory_order_acquire);
    if (s == st_ok || s == st_err) {
        done = true;
        return m.finish(op, s, err);
    }
    return true;
}

bool link::wait(int32_t op, std::string & err, double timeout_s) {
    int32_t which = 0;
    return wait_any(&op, 1, which, err, timeout_s);
}

bool link::wait_any(const int32_t * ops, int32_t n, int32_t & which, std::string & err, double timeout_s) {
    impl & m = *p_;
    which    = -1;
    if (n < 1 || n > 8) {
        return fail(err, "link: wait_any of 1 .. 8 operations");
    }
    for (int32_t k = 0; k < n; ++k) {
        if (ops[k] < 0 || ops[k] >= n_ops) {
            return fail(err, "link: bad operation " + std::to_string(ops[k]));
        }
    }
    const double    bound = timeout_s < 0 ? m.o.timeout_s : timeout_s;
    const auto      t0    = clk::now();
    int64_t         spins = 0;
    clk::time_point lost_at;
    bool            saw_lost = false;
    for (;;) {
        for (int32_t k = 0; k < n; ++k) {
            const int32_t s = m.ops[ops[k]].state.load(std::memory_order_acquire);
            if (s == st_ok || s == st_err) {
                which = k;
                m.tot.wait_s += since(t0);
                return m.finish(ops[k], s, err);
            }
        }
        if (m.lost_flag.load(std::memory_order_acquire)) {
            // the requests complete with errors shortly after the loss; give them a moment, then end the wait
            if (!saw_lost) {
                saw_lost = true;
                lost_at  = clk::now();
            } else if (since(lost_at) > 0.5) {
                which = 0;
                err   = "split stage lost: " + tag_str(m.ops[ops[0]].tag) + ": " + m.lost_why();
                for (int32_t k = 0; k < n; ++k) {
                    m.abandon(ops[k]);
                }
                m.tot.wait_s += since(t0);
                return false;
            }
        }
        if ((++spins & 1023) == 0 && since(t0) > bound) {
            which       = 0;
            err         = "no completion in " + std::to_string(bound) + " s: " + tag_str(m.ops[ops[0]].tag);
            uint64_t ut = 0;
            size_t   ub = 0;
            if (unexpected(ut, ub)) {
                err += " (refused: an unexpected " + tag_str(ut) + " of " + std::to_string(ub) + " bytes is waiting)";
            }
            for (int32_t k = 0; k < n; ++k) {
                m.abandon(ops[k]);
            }
            m.tot.wait_s += since(t0);
            return false;
        }
        relax();
    }
}

bool link::unexpected(uint64_t & tag, size_t & bytes) {
    impl & m     = *p_;
    bool   found = false;
    m.run_job([&]() {
        ucp_tag_recv_info_t info = {};
        ucp_tag_message_h   h    = ucp_tag_probe_nb(m.worker, 0, 0, 0, &info);
        if (h != nullptr) {
            found = true;
            tag   = info.sender_tag;
            bytes = info.length;
        }
    });
    return found;
}

void link::stop() {
    impl & m = *p_;
    if (m.running.load()) {
        while (!m.push(sub(sub_stop, 0))) {
            relax();
        }
        if (m.th.joinable()) {
            m.th.join();
        }
        m.running.store(false);
    }
    if (m.worker != nullptr) {
        // requests still posted end with the worker; their slots are not reused after stop
        for (const impl::mapped & x : m.maps) {
            ucp_mem_unmap(m.ctx, x.h);
        }
        m.maps.clear();
        if (m.listener != nullptr) {
            ucp_listener_destroy(m.listener);
            m.listener = nullptr;
        }
        ucp_worker_destroy(m.worker);
        m.worker = nullptr;
    }
    if (m.ctx != nullptr) {
        ucp_cleanup(m.ctx);
        m.ctx = nullptr;
    }
    for (int * fd : { &m.evfd, &m.epfd }) {
        if (*fd >= 0) {
            close(*fd);
            *fd = -1;
        }
    }
}

}  // namespace fn::xchg::link
