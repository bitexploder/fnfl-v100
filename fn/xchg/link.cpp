#include "fn/xchg/link.h"

#include "fn/xchg/life.h"
#include "fn/xchg/link_fault.h"
#include "fn/xchg/slow.h"

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
#include <deque>
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
// TTTS7: the link thread keeps polling this long after its last activity before it sleeps, so a post or an arrival
// between a serving round's messages finds it awake (no eventfd write, no epoll wake-up); about the length of a round's
// stage (TTTS4 measured about 19 ms at U = 4) [estimated], so a serving split never sleeps and an idle one does
constexpr double    idle_spin_s = 0.025;

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

// TTTSBS: an operation slot's word: its use's generation (20 bits) above its state (8 bits), one atomic, so every
// hand-off between the runtime and the link thread is one acquire/release pair on it and a stale party is refused:
//   free -> queued      the runtime (its fields written first, then the word released; the post entry follows)
//   queued -> posted    the link thread (CAS): it owns the transport request from here
//   queued -> dropped   the runtime (CAS), cancelling before the post: the transport never saw the buffer
//   posted -> ok | err  the link thread, once the transport ended the request (completed, failed or cancelled): the
//                       transport is done with the buffer; the status fields are written before the release
//   ok | err | dropped -> free   the runtime, observing the end (acquire): the buffer and the slot are released
// Only the runtime frees a slot, and only after it acquired a terminal state, so a buffer is never written after the
// runtime gave it back. The runtime's handle carries the generation: a handle of an earlier use is refused, never
// taken for the slot's current operation.
enum op_state : uint32_t { st_free = 0, st_queued = 1, st_posted = 2, st_ok = 3, st_err = 4, st_dropped = 5 };

constexpr uint32_t gen_mask = 0xfffffu;

uint32_t pack(uint32_t gen, uint32_t st) {
    return (gen & gen_mask) << 8 | st;
}

uint32_t gen_of(uint32_t w) {
    return w >> 8;
}

uint32_t st_of(uint32_t w) {
    return w & 0xffu;
}

bool terminal(uint32_t st) {
    return st == st_ok || st == st_err || st == st_dropped;
}

// submission entries: kind in the top 4 bits, the op's generation (20 bits) and its index (8 bits) below, so an entry
// queued for an earlier use of a slot is recognized and skipped
enum sub_kind : uint32_t { sub_post = 1, sub_cancel = 2, sub_job = 3, sub_stop = 4, sub_quiesce = 5 };

static_assert(n_ops <= 256, "an entry and a handle carry the op index in 8 bits");

uint32_t sub(sub_kind k, uint32_t i, uint32_t gen = 0) {
    return (uint32_t(k) << 28) | ((gen & gen_mask) << 8) | (i & 0xffu);
}

// the runtime's handle of a use: its generation above its index (non-negative)
int32_t handle(int32_t i, uint32_t gen) {
    return int32_t((gen & gen_mask) << 8 | uint32_t(i));
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
        std::atomic<uint32_t>       word{ 0 };  // generation | state (above)
        // the runtime's, written before the queued word's release, read by the link thread after its acquire
        bool                        is_send   = false;
        uint64_t                    tag       = 0;
        void *                      p         = nullptr;
        size_t                      n         = 0;
        // the link thread's, written before a terminal word's release, read by the runtime after its acquire
        ucs_status_t                status    = UCS_OK;
        size_t                      got       = 0;
        uint64_t                    got_tag   = 0;
        ucp_tag_recv_info_t         info      = {};
        void *                      req       = nullptr;  // the link thread's only: the transport request while posted
        // the runtime's only
        bool                        abandoned = false;  // given up by a failed wait (retire): reaped once ended
        std::shared_ptr<const void> keep;               // the buffer's owner, held until the end is observed
        uint64_t                    life = 0;  // the lifetime ledger's user (fn/xchg/life.h; 0 outside a test build)
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

    std::vector<mapped> maps;                   // link thread only once running (map runs as a job)
    counters            tot;
    hello               hello_mine, hello_got;  // X-HELLO's buffers: the link's own (a failed exchange's may linger)
    bool                quiesce_asked = false;  // the runtime's: one quiesce entry

#if defined(FNFL_TEST_HOOKS)
    // fn/xchg/link_fault.h: the in-process network in place of UCX, and the hooks
    std::shared_ptr<fault::net> fk;
    int32_t                     side = 0;
    std::atomic<bool>           hold{ false }, parked{ false };
    int32_t                     fail_nth = 0;  // the runtime's: the n-th submission from now is refused
    std::function<void()>       cancel_hook;   // set before the pair starts or while the thread is parked
    void                        fake_post(int32_t i);
    void                        fake_cancel(int32_t i);
    void                        fake_close();
    int32_t                     fake_progress();
#endif

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

    // the link thread: a posted op's end published (its word was posted; only this thread moves it on)
    static void complete(op & x, ucs_status_t st) {
        x.status = st;
        bool ok  = st == UCS_OK;
        if (ok && !x.is_send) {
            x.got     = x.info.length;
            x.got_tag = x.info.sender_tag;
            ok        = x.got == x.n;
        }
        x.req            = nullptr;
        const uint32_t g = gen_of(x.word.load(std::memory_order_relaxed));
        x.word.store(pack(g, ok ? st_ok : st_err), std::memory_order_release);
    }

    static void send_cb(void * req, ucs_status_t st, void * user) {
        complete(*static_cast<op *>(user), st);
        ucp_request_free(req);
    }

    static void recv_cb(void * req, ucs_status_t st, const ucp_tag_recv_info_t * info, void * user) {
        op & x = *static_cast<op *>(user);
        if (info != nullptr) {
            x.info = *info;
        }
        complete(x, st);
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

    void post(int32_t i, uint32_t gen) {
        op &                x   = ops[i];
        ucp_request_param_t prm = {};
        prm.op_attr_mask        = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
        prm.user_data           = &x;
        if (ucp_mem_h h = memh_of(x.p, x.n)) {
            prm.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
            prm.memh = h;
        }
        ucs_status_ptr_t r      = nullptr;
        uint32_t         expect = pack(gen, st_queued);
        if (!x.word.compare_exchange_strong(expect, pack(gen, st_posted), std::memory_order_acq_rel)) {
            return;  // dropped by the runtime before its post, or a stale entry
        }
#if defined(FNFL_TEST_HOOKS)
        if (fk != nullptr) {
            if (lost_flag.load(std::memory_order_acquire)) {
                complete(x, UCS_ERR_CONNECTION_RESET);
            } else {
                fake_post(i);
            }
            return;
        }
#endif
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
            if (st_of(x.word.load(std::memory_order_relaxed)) == st_posted) {
                return true;
            }
        }
        return false;
    }

    // the link thread: every posted request cancelled and the endpoint closed (forced: the transport stops touching
    // memory; a lost peer's requests end with errors), then progress until no request is posted, bounded. The link
    // is lost after it (no new operation)
    void quiesce_now(const char * why_now) {
        {
            std::lock_guard<std::mutex> g(why_mu);
            if (why.empty()) {
                why = why_now;
            }
        }
        lost_flag.store(true, std::memory_order_release);
#if defined(FNFL_TEST_HOOKS)
        if (fk != nullptr) {
            fake_close();
            fake_progress();
            return;
        }
#endif
        for (op & x : ops) {
            if (st_of(x.word.load(std::memory_order_relaxed)) == st_posted && x.req != nullptr) {
                ucp_request_cancel(worker, x.req);
            }
        }
        close_ep(true);
        const auto t0 = clk::now();
        while (any_posted() && since(t0) < o.retire_s) {
            ucp_worker_progress(worker);
        }
    }

    // the link thread: the endpoint closed, bounded (forced: no flush)
    void close_ep(bool force) {
        if (ep == nullptr) {
            return;
        }
        ucp_request_param_t prm = {};
        prm.op_attr_mask        = UCP_OP_ATTR_FIELD_FLAGS;
        prm.flags               = force ? UCP_EP_CLOSE_FLAG_FORCE : 0;
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

    void run() {
        epoll_event ev[2];
        auto        busy = clk::now();  // the last activity: an entry taken, progress, or a request outstanding
        while (!stopping.load(std::memory_order_acquire)) {
#if defined(FNFL_TEST_HOOKS)
            if (hold.load(std::memory_order_acquire)) {
                parked.store(true, std::memory_order_release);
                while (hold.load(std::memory_order_acquire) && !stopping.load(std::memory_order_acquire)) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                parked.store(false, std::memory_order_release);
                continue;
            }
#endif
            uint32_t h = q_head.load(std::memory_order_acquire);
            uint32_t t = q_tail.load(std::memory_order_relaxed);
            if (t != h) {
                busy = clk::now();
            }
            while (t != h) {
                const uint32_t e = q[t & (q_cap - 1)];
                t++;
                q_tail.store(t, std::memory_order_release);
                const uint32_t k = e >> 28, i = e & 0xffu, gen = (e >> 8) & 0xfffffu;
                if (k == sub_post) {
                    post(int32_t(i), gen);
                } else if (k == sub_cancel) {
#if defined(FNFL_TEST_HOOKS)
                    if (cancel_hook) {
                        cancel_hook();
                        if (fk != nullptr) {
                            fake_progress();  // what the hook released completes before the cancel is acted on
                        }
                    }
#endif
                    op & x = ops[i];
                    // only a posted request of this use is cancelled (an ended one, or a stale entry: nothing);
                    // the request's end (completed or cancelled) is published by its callback as any end
                    if (x.word.load(std::memory_order_acquire) == pack(gen, st_posted) && x.req != nullptr) {
#if defined(FNFL_TEST_HOOKS)
                        if (fk != nullptr) {
                            fake_cancel(int32_t(i));
                        } else
#endif
                        {
                            ucp_request_cancel(worker, x.req);
                        }
                    }
                } else if (k == sub_job) {
                    job();
                    job_done.store(true, std::memory_order_release);
                } else if (k == sub_quiesce) {
                    quiesce_now("the link was quiesced (an operation's end could not be observed)");
                } else if (k == sub_stop) {
                    stopping.store(true, std::memory_order_release);
                }
                h = q_head.load(std::memory_order_acquire);
            }
            int progressed = 0;
#if defined(FNFL_TEST_HOOKS)
            if (fk != nullptr) {
                progressed = fake_progress();
                if (progressed == 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(20));
                }
                continue;
            }
#endif
            while (ucp_worker_progress(worker) != 0) {
                ++progressed;
            }
            if (progressed > 0 || any_posted()) {
                busy = clk::now();
                relax();
                continue;
            }
            if (since(busy) < idle_spin_s) {
                relax();
                continue;
            }
            // idle for idle_spin_s: arm the worker, publish that this thread sleeps, recheck the queue, then sleep
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
        // stopping (TTTSBS): every operation still posted ends before the thread does: the receives cancelled, the
        // endpoint flushed and closed (forced when lost, or when a flush leaves requests), bounded; stop() then
        // destroys the worker and only then unmaps
#if defined(FNFL_TEST_HOOKS)
        if (fk != nullptr) {
            fake_close();
            fake_progress();
            return;
        }
#endif
        for (op & x : ops) {
            if (st_of(x.word.load(std::memory_order_relaxed)) == st_posted && x.req != nullptr && !x.is_send) {
                ucp_request_cancel(worker, x.req);
            }
        }
        close_ep(lost_flag.load(std::memory_order_acquire));
        const auto t0 = clk::now();
        while (any_posted() && since(t0) < o.retire_s) {
            ucp_worker_progress(worker);
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

    // an abandoned op whose end is now visible: released without a result (its owner's reference dropped)
    void reap(int32_t i) {
        op &           x = ops[i];
        const uint32_t w = x.word.load(std::memory_order_acquire);
        if (x.abandoned && terminal(st_of(w))) {
            life::done(x.life, st_of(w) == st_dropped ? "dropped before its post" : "ended after it was given up");
            x.life      = 0;
            x.abandoned = false;
            x.keep.reset();
            x.word.store(pack(gen_of(w), st_free), std::memory_order_release);
        }
    }

    int32_t alloc(std::string & err) {
        for (int32_t i = 0; i < n_ops; ++i) {
            reap(i);
            if (st_of(ops[i].word.load(std::memory_order_acquire)) == st_free) {
                return i;
            }
        }
        err = "link: no free operation slot (" + std::to_string(n_ops) + " in flight)";
        return -1;
    }

    int32_t submit(bool                        is_send,
                   uint64_t                    tag,
                   void *                      p,
                   size_t                      n,
                   std::shared_ptr<const void> keep,
                   std::string &               err) {
        if (lost_flag.load(std::memory_order_acquire)) {
            err = "split stage lost: " + tag_str(tag) + ": " + lost_why();
            return -1;
        }
#if defined(FNFL_TEST_HOOKS)
        if (fail_nth > 0 && --fail_nth == 0) {
            err = "link: TEST: submission refused (" + tag_str(tag) + ")";
            return -1;
        }
#endif
        const int32_t i = alloc(err);
        if (i < 0) {
            return -1;
        }
        op &           x   = ops[i];
        const uint32_t gen = (gen_of(x.word.load(std::memory_order_relaxed)) + 1) & gen_mask;
        x.is_send          = is_send;
        x.tag              = tag;
        x.p                = p;
        x.n                = n;
        x.status           = UCS_OK;
        x.got              = 0;
        x.got_tag          = 0;
        x.info             = {};
        x.abandoned        = false;
        x.keep             = std::move(keep);
        if constexpr (life::on) {
            x.life = life::next_id();
            life::use(x.life, p, n, is_send ? "link send" : "link receive");
        }
        x.word.store(pack(gen, st_queued), std::memory_order_release);
        if (!push(sub(sub_post, uint32_t(i), gen))) {
            life::done(x.life, "never queued");
            x.life = 0;
            x.keep.reset();
            x.word.store(pack(gen, st_free), std::memory_order_release);
            err = "link: the submission queue is full";
            return -1;
        }
        return handle(i, gen);
    }

    std::string lost_why() {
        std::lock_guard<std::mutex> g(why_mu);
        return why.empty() ? std::string("the peer is lost") : why;
    }

    // the op of a handle when it is the slot's current use and not given up; nullptr with err otherwise
    op * of(int32_t h, std::string & err) {
        if (h < 0) {
            err = "link: bad operation " + std::to_string(h);
            return nullptr;
        }
        const int32_t  i = h & 0xff;
        const uint32_t g = uint32_t(h) >> 8;
        op &           x = ops[i];
        const uint32_t w = x.word.load(std::memory_order_acquire);
        if (gen_of(w) != g || st_of(w) == st_free || x.abandoned) {
            err = "link: a stale operation handle " + std::to_string(h) + " (its slot " + std::to_string(i) +
                  " is finished or reused)";
            return nullptr;
        }
        return &x;
    }

    // the outcome of an op whose terminal state the runtime acquired, then the slot freed (its owner released)
    bool finish(op & x, uint32_t w, std::string & err) {
        const uint32_t s  = st_of(w);
        bool           ok = s == st_ok;
        if (!ok) {
            if (s == st_dropped) {
                err = "cancelled before its post: " + tag_str(x.tag);
            } else if (x.status == UCS_OK && !x.is_send) {
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
        life::done(x.life, ok ? "completed" : "failed");
        x.life      = 0;
        x.abandoned = false;
        x.keep.reset();
        x.word.store(pack(gen_of(w), st_free), std::memory_order_release);
        return ok;
    }

    // give up an op: before its post, dropped here (the transport never saw it); after, its cancel queued. Its end
    // is observed later (retire, reap)
    void abandon(op & x, int32_t i) {
        x.abandoned           = true;
        const uint32_t w      = x.word.load(std::memory_order_acquire);
        uint32_t       expect = pack(gen_of(w), st_queued);
        if (x.word.compare_exchange_strong(expect, pack(gen_of(w), st_dropped), std::memory_order_acq_rel)) {
            return;  // its post entry finds it dropped and skips it
        }
        if (st_of(expect) == st_posted) {
            while (!push(sub(sub_cancel, uint32_t(i), gen_of(w)))) {
                relax();
            }
        }
    }

    void quiesce(const std::string & why_now) {
        if (!running.load()) {
            return;
        }
        {
            std::lock_guard<std::mutex> g(why_mu);
            if (why.empty()) {
                why = why_now;
            }
        }
        if (!quiesce_asked) {
            quiesce_asked = true;
            while (!push(sub(sub_quiesce, 0))) {
                relax();
            }
        }
    }

    // TTTSBS: end every op of hs (handles; -1 and stale ones skipped): each cancelled, its end observed within
    // retire_s, else the link quiesced and its ends awaited once more; then each ended op finished (ok_out: the one
    // that completed with its bytes, when ok_out names it). Returns how many could not be observed ended: those stay
    // given up with their owners held (reaped once they end, or at stop)
    int32_t retire(const int32_t * hs,
                   int32_t         n,
                   int32_t         ok_of  = -1,
                   bool *          ok_out = nullptr,
                   std::string *   err    = nullptr) {
        std::vector<int32_t>  idx;
        std::vector<uint32_t> gens;
        for (int32_t k = 0; k < n; ++k) {
            std::string e;
            op *        x = hs[k] < 0 ? nullptr : of(hs[k], e);
            idx.push_back(x == nullptr ? -1 : int32_t(hs[k] & 0xff));
            gens.push_back(x == nullptr ? 0 : uint32_t(hs[k]) >> 8);
            if (x != nullptr && !terminal(st_of(x->word.load(std::memory_order_acquire)))) {
                abandon(*x, idx.back());
            }
        }
        const auto ended = [&]() {
            for (int32_t i : idx) {
                if (i >= 0 && !terminal(st_of(ops[i].word.load(std::memory_order_acquire)))) {
                    return false;
                }
            }
            return true;
        };
        const auto await = [&](double bound) {
            const auto t0 = clk::now();
            while (!ended() && since(t0) < bound) {
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
        };
        await(o.retire_s);
        if (!ended()) {
            quiesce("split stage lost: an operation's end was not observed within " + std::to_string(o.retire_s) +
                    " s of its cancel; the link was quiesced");
            await(o.retire_s);
        }
        int32_t held = 0;
        for (size_t k = 0; k < idx.size(); ++k) {
            if (idx[k] < 0) {
                continue;
            }
            op &           x = ops[idx[k]];
            const uint32_t w = x.word.load(std::memory_order_acquire);
            if (!terminal(st_of(w))) {
                x.abandoned = true;
                ++held;
                continue;
            }
            std::string e;
            const bool  ok = finish(x, w, e);
            if (int32_t(k) == ok_of && ok_out != nullptr) {
                *ok_out = ok;
                if (!ok && err != nullptr && err->empty()) {
                    *err = e;
                }
            }
        }
        return held;
    }

    // given-up ops not ended (each reaped first when it has)
    int32_t held() {
        int32_t h = 0;
        for (int32_t i = 0; i < n_ops; ++i) {
            reap(i);
            h += ops[i].abandoned ? 1 : 0;
        }
        return h;
    }
};

#if defined(FNFL_TEST_HOOKS)
bool fault_unexpected(link::impl & m, uint64_t & tag, size_t & bytes);
#endif

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

int32_t link::held() {
    return p_->held();
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
        // TTTSBS: its buffers are the link's (hello_mine, hello_got), so a request a failed attempt leaves cannot
        // outlive them; a request that does not end after its cancel refuses the start
        const auto exchange = [&](std::string & e) -> bool {
            hello & mine   = m.hello_mine;
            hello & got    = m.hello_got;
            mine           = m.o.me;
            got            = hello();
            impl::op & rs  = m.ops[0];
            impl::op & ss  = m.ops[1];
            const auto arm = [&](impl::op & x, int32_t i, bool snd, void * p) {
                x.is_send        = snd;
                x.tag            = make_tag(kind::hello, 0, 0, 0);
                x.p              = p;
                x.n              = sizeof(hello);
                x.status         = UCS_OK;
                x.info           = {};
                const uint32_t g = (gen_of(x.word.load()) + 1) & gen_mask;
                x.word.store(pack(g, st_queued));
                m.post(i, g);
            };
            const auto st_now = [](impl::op & x) {
                return st_of(x.word.load());
            };
            const auto unarm = [](impl::op & x) {
                x.word.store(pack(gen_of(x.word.load()), st_free));
            };
            arm(rs, 0, false, &got);
            if (o.leader) {
                arm(ss, 1, true, &mine);
            }
            bool sent = o.leader;  // the leader sent its X-HELLO above; the follower answers once it has the leader's
            const auto tw = clk::now();
            while (since(tw) < std::max(2.0, o.timeout_s)) {
                ucp_worker_progress(m.worker);
                if (m.lost_flag.load()) {
                    break;
                }
                if (!o.leader && st_now(rs) == st_ok && !sent) {
                    arm(ss, 1, true, &mine);  // the follower answers after it has read the leader's
                    sent = true;
                }
                const uint32_t a = st_now(rs), b = st_now(ss);
                if ((a == st_ok || a == st_err) && sent && (b == st_ok || b == st_err)) {
                    break;
                }
            }
            const bool ok = st_now(rs) == st_ok && st_now(ss) == st_ok;
            if (!ok) {
                const auto t1 = clk::now();
                if (rs.req != nullptr) {
                    ucp_request_cancel(m.worker, rs.req);
                }
                while ((rs.req != nullptr || ss.req != nullptr) && since(t1) < std::max(2.0, o.retire_s)) {
                    ucp_worker_progress(m.worker);
                }
                if (rs.req != nullptr || ss.req != nullptr) {
                    return fail(e, "refused to start: an X-HELLO request did not end after its cancel");
                }
                unarm(rs);
                unarm(ss);
                return fail(e, m.lost_flag.load() ? m.lost_why() : "no X-HELLO answer");
            }
            unarm(rs);
            unarm(ss);
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
#if defined(FNFL_TEST_HOOKS)
        if (m.fk != nullptr) {
            m.maps.push_back({ static_cast<uint8_t *>(p), n, nullptr });
            life::map(p, n);
            return;
        }
#endif
        st = ucp_mem_map(m.ctx, &mp, &h);
        if (st == UCS_OK) {
            m.maps.push_back({ static_cast<uint8_t *>(p), n, h });
            life::map(p, n);
        }
    });
    if (st != UCS_OK) {
        return fail(err, std::string("link: ucp_mem_map of ") + std::to_string(n) + " bytes: " + ucs_status_string(st));
    }
    return true;
}

int32_t link::send(uint64_t tag, const void * p, size_t n, std::string & err, std::shared_ptr<const void> keep) {
    return p_->submit(true, tag, const_cast<void *>(p), n, std::move(keep), err);
}

int32_t link::recv(uint64_t tag, void * p, size_t n, std::string & err, std::shared_ptr<const void> keep) {
    return p_->submit(false, tag, p, n, std::move(keep), err);
}

bool link::test(int32_t op, bool & done, std::string & err) {
    impl & m = *p_;
    done     = false;
    impl::op * x = m.of(op, err);
    if (x == nullptr) {
        done = true;  // a stale handle: nothing of it is left to wait for
        return false;
    }
    const uint32_t w = x->word.load(std::memory_order_acquire);
    if (terminal(st_of(w))) {
        done = true;
        return m.finish(*x, w, err);
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
    impl::op * x[8] = {};
    for (int32_t k = 0; k < n; ++k) {
        x[k] = m.of(ops[k], err);
        if (x[k] == nullptr) {
            return false;  // a stale handle: refused, the slot's current use untouched
        }
    }
    const double    bound = timeout_s < 0 ? m.o.timeout_s : timeout_s;
    const uint64_t  tag0  = x[0]->tag;  // the give-up's line names it (retire releases the slot)
    const auto      t0    = clk::now();
    int64_t         spins = 0;
    clk::time_point lost_at;
    bool            saw_lost = false;
    // TTTSBS: a wait that gives up retires its operations (each cancelled and its end observed) before it returns,
    // so the caller may release or reuse their buffers; a completion that won the race with the cancel is the
    // wait's success. One whose end cannot be observed is held by the link with its owner (retire)
    // SSF1: a wait of slow::threshold_s or more while writers generate is a trace line (its tag and how it ended)
    const auto      report   = [&](const char * how, uint64_t tag) {
        const double s = since(t0);
        m.tot.wait_s += s;
        if (slow::due(s)) {
            slow::line(
                "link wait", s,
                std::string(how) + ": " + tag_str(tag) + " (" + std::to_string(n) + " op" + (n == 1 ? "" : "s") + ")");
        }
    };
    const auto give_up = [&](const char * how) {
        bool          ok = false;
        std::string   e2;
        const int32_t held = m.retire(ops, n, n == 1 ? 0 : -1, &ok, &e2);
        if (n == 1 && ok) {
            report("completed after a give-up", tag0);
            err.clear();
            which = 0;
            return true;
        }
        report(how, tag0);
        if (held > 0) {
            err += " (" + std::to_string(held) + " operation" + (held == 1 ? "" : "s") +
                   " given up whose end was not observed: held, the link quiesced)";
        }
        return false;
    };
    for (;;) {
        for (int32_t k = 0; k < n; ++k) {
            const uint32_t w = x[k]->word.load(std::memory_order_acquire);
            if (terminal(st_of(w))) {
                which              = k;
                const uint64_t tag = x[k]->tag;  // before finish releases the slot
                const bool     ok  = m.finish(*x[k], w, err);
                report(ok ? "completed" : "failed", tag);
                return ok;
            }
        }
        if (m.lost_flag.load(std::memory_order_acquire)) {
            // the requests complete with errors shortly after the loss; give them a moment, then end the wait
            if (!saw_lost) {
                saw_lost = true;
                lost_at  = clk::now();
            } else if (since(lost_at) > 0.5) {
                which = 0;
                err   = "split stage lost: " + tag_str(x[0]->tag) + ": " + m.lost_why();
                return give_up("lost");
            }
        }
        if ((++spins & 1023) == 0 && since(t0) > bound) {
            which       = 0;
            err         = "no completion in " + std::to_string(bound) + " s: " + tag_str(x[0]->tag);
            uint64_t ut = 0;
            size_t   ub = 0;
            if (unexpected(ut, ub)) {
                err += " (refused: an unexpected " + tag_str(ut) + " of " + std::to_string(ub) + " bytes is waiting)";
            }
            return give_up("timed out");
        }
        relax();
    }
}

int32_t link::retire(const int32_t * ops, int32_t n) {
    return p_->retire(ops, n);
}

void link::quiesce(const std::string & why) {
    p_->quiesce(why);
}

bool link::unexpected(uint64_t & tag, size_t & bytes) {
    impl & m     = *p_;
    bool   found = false;
#if defined(FNFL_TEST_HOOKS)
    if (m.fk != nullptr) {
        return fault_unexpected(m, tag, bytes);  // the test network's probe (no job: its thread may be held)
    }
#endif
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
    // TTTSBS: the link thread ended what it could before it returned; every op that ended is released now, and one
    // still posted ends with the worker, destroyed before any range is unmapped and before any owner is released
    const auto release_ended = [&]() {
        for (impl::op & x : m.ops) {
            const uint32_t w = x.word.load(std::memory_order_acquire);
            if (terminal(st_of(w)) || st_of(w) == st_queued) {  // queued: never posted (the thread is gone)
                life::done(x.life, "ended at stop");
                x.life      = 0;
                x.abandoned = false;
                x.keep.reset();
                x.word.store(pack(gen_of(w), st_free), std::memory_order_release);
            }
        }
    };
    const auto release_rest = [&](const char * how) {
        for (impl::op & x : m.ops) {
            const uint32_t w = x.word.load(std::memory_order_acquire);
            if (st_of(w) != st_free) {
                life::done(x.life, how);
                x.life      = 0;
                x.abandoned = false;
                x.req       = nullptr;
                x.keep.reset();
                x.word.store(pack(gen_of(w), st_free), std::memory_order_release);
            }
        }
    };
    release_ended();
#if defined(FNFL_TEST_HOOKS)
    if (m.fk != nullptr) {
        m.fk.reset();  // no thread progresses the test network now: nothing is copied after this
        release_rest("ended with the test network");
        for (const impl::mapped & x : m.maps) {
            life::unmap(x.p, x.n);
        }
        m.maps.clear();
        return;
    }
#endif
    if (m.worker != nullptr) {
        if (m.listener != nullptr) {
            ucp_listener_destroy(m.listener);
            m.listener = nullptr;
        }
        ucp_worker_destroy(m.worker);
        m.worker = nullptr;
    }
    release_rest("ended with the worker");
    for (const impl::mapped & x : m.maps) {
        life::unmap(x.p, x.n);
        ucp_mem_unmap(m.ctx, x.h);
    }
    m.maps.clear();
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

#if defined(FNFL_TEST_HOOKS)

// ---- the test hooks' in-process network (fn/xchg/link_fault.h) ----

struct fault::net {
    struct msg {  // a posted send, sent to the other side
        int32_t      op  = 0;
        uint32_t     gen = 0;
        uint64_t     tag = 0;
        const void * p   = nullptr;
        size_t       n   = 0;
    };

    struct rcv {  // a posted receive
        int32_t  op   = 0;
        uint32_t gen  = 0;
        uint64_t tag  = 0;
        void *   p    = nullptr;
        size_t   n    = 0;
        uint64_t life = 0;
    };

    struct xfer {  // matched, its copy not run yet (run by the receiver's link thread)
        rcv  r;
        msg  m;
        bool held = false, stuck = false;
    };

    struct fin {  // an end for a side's own operation, published by that side's link thread
        int32_t      op  = 0;
        uint32_t     gen = 0;
        ucs_status_t st  = UCS_OK;
        size_t       got = 0;
        uint64_t     tag = 0;
    };

    struct gate {
        uint64_t mask = 0, value = 0;
        bool     stuck = false;
    };

    std::mutex        mu;
    link::impl *      side[2] = { nullptr, nullptr };
    std::deque<msg>   incoming[2];  // sent to side s, no receive matched yet (unexpected)
    std::deque<rcv>   posted[2];    // side s's receives, not matched yet
    std::deque<xfer>  matched[2];   // side s's matched receives, copy pending
    std::deque<fin>   done[2];      // side s's ends to publish
    std::vector<gate> gates[2];     // side s's holds on its receives
    bool              lost_seen[2] = { false, false };
    bool              lost         = false;
    int64_t           copies[2] = { 0, 0 }, dropped[2] = { 0, 0 };

    bool gated(int32_t s, uint64_t tag, bool & stuck) const {
        for (const gate & g : gates[s]) {
            if ((tag & g.mask) == g.value) {
                stuck = g.stuck;
                return true;
            }
        }
        return false;
    }

    // every operation of side s not stuck ends with st (a forced close, a lost peer); the peer's halves of side s's
    // transfers end with a reset. Under mu
    void end_side(int32_t s, ucs_status_t st) {
        const int32_t o = 1 - s;
        for (const rcv & r : posted[s]) {
            done[s].push_back({ r.op, r.gen, st, 0, 0 });
        }
        posted[s].clear();
        for (auto it = incoming[o].begin(); it != incoming[o].end();) {  // s's unmatched sends
            done[s].push_back({ it->op, it->gen, st, 0, 0 });
            it = incoming[o].erase(it);
        }
        for (int32_t q = 0; q < 2; ++q) {
            for (auto it = matched[q].begin(); it != matched[q].end();) {
                if (it->stuck) {
                    ++it;
                    continue;
                }
                // a transfer into side q from side 1 - q: both halves end, no copy
                done[q].push_back({ it->r.op, it->r.gen, q == s ? st : UCS_ERR_CONNECTION_RESET, 0, 0 });
                done[1 - q].push_back({ it->m.op, it->m.gen, 1 - q == s ? st : UCS_ERR_CONNECTION_RESET, 0, 0 });
                ++dropped[q];
                it = matched[q].erase(it);
            }
        }
    }
};

struct fault_access {
    static link::impl & of(link & l) { return *l.p_; }
};

void link::impl::fake_post(int32_t i) {
    op &                        x = ops[i];
    std::lock_guard<std::mutex> g(fk->mu);
    if (x.is_send) {
        fault::net::msg m;
        m.op  = i;
        m.gen = gen_of(x.word.load(std::memory_order_relaxed));
        m.tag = x.tag;
        m.p   = x.p;
        m.n   = x.n;
        fk->incoming[1 - side].push_back(m);
    } else {
        fault::net::rcv r;
        r.op   = i;
        r.gen  = gen_of(x.word.load(std::memory_order_relaxed));
        r.tag  = x.tag;
        r.p    = x.p;
        r.n    = x.n;
        r.life = x.life;
        fk->posted[side].push_back(r);
    }
    x.req = &x;  // a request outstanding (the cancel path's test); never freed
}

void link::impl::fake_cancel(int32_t i) {
    op &                        x = ops[i];
    std::lock_guard<std::mutex> g(fk->mu);
    auto &                      pr = fk->posted[side];
    const uint32_t              gn = gen_of(x.word.load(std::memory_order_relaxed));
    for (auto it = pr.begin(); it != pr.end(); ++it) {
        if (it->op == i && it->gen == gn) {
            fk->done[side].push_back({ i, gn, UCS_ERR_CANCELED, 0, 0 });
            pr.erase(it);
            return;
        }
    }
    // matched (its copy pending) or a send: not cancellable, as UCP's in-progress receives and sends
}

void link::impl::fake_close() {
    std::lock_guard<std::mutex> g(fk->mu);
    fk->lost = true;
    fk->end_side(side, UCS_ERR_CANCELED);
    fk->end_side(1 - side, UCS_ERR_CONNECTION_RESET);
}

int32_t link::impl::fake_progress() {
    std::vector<fault::net::fin> fins;
    bool                         lost_now = false;
    {
        std::lock_guard<std::mutex> g(fk->mu);
        fault::net &                f = *fk;
        const int32_t               s = side;
        if (f.lost && !f.lost_seen[s]) {
            f.lost_seen[s] = true;
            lost_now       = true;
        }
        // match: messages in arrival order, each to the first posted receive of its tag
        for (auto mi = f.incoming[s].begin(); mi != f.incoming[s].end();) {
            auto ri = f.posted[s].begin();
            while (ri != f.posted[s].end() && ri->tag != mi->tag) {
                ++ri;
            }
            if (ri == f.posted[s].end()) {
                ++mi;
                continue;
            }
            fault::net::xfer x;
            x.r    = *ri;
            x.m    = *mi;
            x.held = f.gated(s, x.r.tag, x.stuck);
            f.matched[s].push_back(x);
            f.posted[s].erase(ri);
            mi = f.incoming[s].erase(mi);
        }
        // run: the delayed copy, reading the sender's buffer now
        for (auto xi = f.matched[s].begin(); xi != f.matched[s].end();) {
            if (xi->held) {
                ++xi;
                continue;
            }
            const size_t c = std::min(xi->r.n, xi->m.n);
            life::publish(xi->r.life);
            std::memcpy(xi->r.p, xi->m.p, c);
            f.copies[s] += int64_t(c);
            const ucs_status_t rs = xi->m.n > xi->r.n ? UCS_ERR_MESSAGE_TRUNCATED : UCS_OK;
            f.done[s].push_back({ xi->r.op, xi->r.gen, rs, xi->m.n, xi->m.tag });
            f.done[1 - s].push_back({ xi->m.op, xi->m.gen, UCS_OK, 0, 0 });
            xi = f.matched[s].erase(xi);
        }
        fins.assign(f.done[s].begin(), f.done[s].end());
        f.done[s].clear();
    }
    if (lost_now) {
        {
            std::lock_guard<std::mutex> g(why_mu);
            if (why.empty()) {
                why = "the peer is lost (TEST: killed)";
            }
        }
        lost_flag.store(true, std::memory_order_release);
    }
    for (const fault::net::fin & e : fins) {
        op &           x = ops[e.op];
        const uint32_t w = x.word.load(std::memory_order_relaxed);
        if (w != pack(e.gen, st_posted) || x.req == nullptr) {
            fprintf(stderr, "link: TEST: a stale end for operation %d (generation %u, word %#x) dropped\n", e.op, e.gen,
                    w);
            continue;
        }
        if (!x.is_send) {
            x.info.length     = e.got;
            x.info.sender_tag = e.tag;
        }
        complete(x, e.st);
    }
    return int32_t(fins.size());
}

bool fault_unexpected(link::impl & m, uint64_t & tag, size_t & bytes) {
    std::lock_guard<std::mutex> g(m.fk->mu);
    const auto &                in = m.fk->incoming[m.side];
    if (in.empty()) {
        return false;
    }
    tag   = in.front().tag;
    bytes = in.front().n;
    return true;
}

namespace fault {

bool pair(link & a, link & b, const options & oa, const options & ob, std::string & err) {
    auto            f     = std::make_shared<net>();
    link::impl *    im[2] = { &fault_access::of(a), &fault_access::of(b) };
    const options * o[2]  = { &oa, &ob };
    for (int32_t s = 0; s < 2; ++s) {
        link::impl & m = *im[s];
        if (m.running.load()) {
            err = "link: TEST: pair of a started link";
            return false;
        }
        m.o         = *o[s];
        m.o.leader  = s == 0;
        m.o.me.role = s;
        m.fk        = f;
        m.side      = s;
        m.transport = "test/in-process";
        f->side[s]  = &m;
    }
    im[0]->theirs = im[1]->o.me;
    im[1]->theirs = im[0]->o.me;
    for (int32_t s = 0; s < 2; ++s) {
        link::impl & m = *im[s];
        m.running.store(true);
        m.th = std::thread([&m]() { m.run(); });
    }
    return true;
}

void hold_transfers(link & receiver, uint64_t mask, uint64_t value, bool stuck) {
    link::impl &                m = fault_access::of(receiver);
    std::lock_guard<std::mutex> g(m.fk->mu);
    m.fk->gates[m.side].push_back({ mask, value, stuck });
}

void release_transfers(link & receiver) {
    link::impl &                m = fault_access::of(receiver);
    std::lock_guard<std::mutex> g(m.fk->mu);
    m.fk->gates[m.side].clear();
    for (net::xfer & x : m.fk->matched[m.side]) {
        x.held  = false;
        x.stuck = false;
    }
}

int32_t held_transfers(link & receiver) {
    link::impl &                m = fault_access::of(receiver);
    std::lock_guard<std::mutex> g(m.fk->mu);
    int32_t                     n = 0;
    for (const net::xfer & x : m.fk->matched[m.side]) {
        n += x.held ? 1 : 0;
    }
    return n;
}

int32_t unexpected_count(link & receiver) {
    link::impl &                m = fault_access::of(receiver);
    std::lock_guard<std::mutex> g(m.fk->mu);
    return int32_t(m.fk->incoming[m.side].size());
}

void hold_thread(link & l, bool on) {
    fault_access::of(l).hold.store(on, std::memory_order_release);
}

void wait_parked(link & l) {
    link::impl & m = fault_access::of(l);
    while (!m.parked.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

void fail_submit(link & l, int32_t nth) {
    fault_access::of(l).fail_nth = nth;
}

void on_cancel(link & l, std::function<void()> f) {
    fault_access::of(l).cancel_hook = std::move(f);
}

void kill(link & l) {
    link::impl &                m = fault_access::of(l);
    std::lock_guard<std::mutex> g(m.fk->mu);
    m.fk->lost = true;
    m.fk->end_side(0, UCS_ERR_CONNECTION_RESET);
    m.fk->end_side(1, UCS_ERR_CONNECTION_RESET);
}

int64_t copies(link & l) {
    link::impl &                m = fault_access::of(l);
    std::lock_guard<std::mutex> g(m.fk->mu);
    return m.fk->copies[m.side];
}

int64_t dropped(link & l) {
    link::impl &                m = fault_access::of(l);
    std::lock_guard<std::mutex> g(m.fk->mu);
    return m.fk->dropped[m.side];
}

}  // namespace fault

#endif

}  // namespace fn::xchg::link
