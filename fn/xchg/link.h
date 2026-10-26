#pragma once

// fn/xchg/link: the box-to-box link of the split (TTTS1): two boxes, tensor parallel inside each, the layers split
// between them. UCX's UCP tagged messages (operator, 2026-10-25: "reuse commms libraries instead of invent"), host
// memory only: every crossing is staged through pinned host buffers, since these boxes have no GPUDirect RDMA. The link
// moves bytes and names them; what the bytes mean (the boundary rows, a step's outcome, a state command) is the
// stage's (fn/runtime/stage.h), so nothing here knows a model constant (the GLM port reuses it).
//
// One context and one worker a box, owned by the link thread: the worker is UCS_THREAD_MODE_SINGLE and only that thread
// progresses it. The thread busy-polls while a request is outstanding and sleeps on the worker's event descriptor (and
// the submission eventfd) when idle. The runtime talks to it through a bounded single-producer submission queue, and
// reads each operation's completion from that operation's own slot (an atomic state the link thread publishes with
// release order), so neither side blocks the other.
//
// Connection: the follower listens (ucp_listener_create) on its address; the leader connects by socket address
// (ucp_ep_create, UCP_EP_PARAM_FIELD_SOCK_ADDR), both in UCP_ERR_HANDLING_MODE_PEER, so a lost peer completes every
// outstanding request with an error. At connect the leader sends X-HELLO and the follower answers with its own: the
// source commit, the device-code digest, the model file's header hash, the split layer k and the slot count must all
// match, or both sides refuse to start (the message names the field).
//
// Transports come from UCX's environment (UCX_TLS, UCX_NET_DEVICES, UCX_IB_GID_INDEX; fnfl-serve.sh --split sets them
// and prints them): rc_x,ud_x,self on mlx5_0:1 for serving, tcp,self for debugging. transport() reports the
// endpoint's lanes as UCX chose them (ucp_ep_query); a serving run must show rc_mlx5, not tcp.
//
// Messages. A tag is kind (8 bits) | group (8) | sequence (32) | card (8) | aux (8), from the top bit down to bit 0;
// a receive matches the full tag (full mask), and its buffer must receive exactly the posted bytes, or the operation
// fails with the tag named (a longer message is UCS_ERR_MESSAGE_TRUNCATED, a shorter one a size mismatch).
//   X-HELLO     (kind hello):    the connect check above, both ways once.
//   X-BOUNDARY  (kind boundary): a decode round's crossing, front to back: one card's boundary rows of one lane (aux:
//                                the lane's slot); a group's crossing is one message a lane and card.
//   X-BOUNDARY  (kind prompt):   a prompt chunk's crossing in pieces (aux: the piece), front to back.
//   X-STEP      (kind step):     a step's outcome and the next round's inputs, back to front, a few hundred bytes.
//   X-CTRL      (kind ctrl):     a state command (request lifecycle, truncate, checkpoints, pool reports, shutdown),
//                                either way; X-ACK (kind ack) answers each, with its sequence.
//   X-BYE       (kind bye):      an orderly shutdown.
//
// The four steps of each crossing (STANDARDS "Memory and publication"), for a buffer B the sender owns and a buffer
// B' the receiver owns, both pinned host memory mapped once (map):
//   1. writes:      the producer fills B: a graph's device-to-host copy node (the copy engine, in stream order before
//                   the round's completion record), or the host for X-STEP and X-CTRL.
//   2. publication: the runtime posts send(tag, B) once the producer's completion is visible on the host (the round's
//                   completion record, acquired); the link thread hands B to UCX.
//   3. acquire:     the receiver's receive into B' was posted before (the receiver keeps the next expected receive
//                   posted); its completion, read with acquire order from the operation's slot, makes B' readable.
//   4. release:     the sender may refill B after its send completes; the receiver may repost B' after its consumer
//                   (a graph's host-to-device copy node, in stream order) has run, or the host has read it.
// Every wait is bounded (options::timeout_s): a wait that times out, or ends because the peer is lost, fails with a
// message that names the kind, group, sequence and card.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace fn::xchg::link {

enum class kind : uint8_t {
    hello    = 1,
    boundary = 2,
    prompt   = 3,
    step     = 4,
    ctrl     = 5,
    ack      = 6,
    bye      = 7,
    test     = 8,                // link-test's own messages
};
const char * kind_name(kind k);  // "X-BOUNDARY", ...

struct tag_fields {
    kind     k     = kind::test;
    uint8_t  group = 0;
    uint32_t seq   = 0;
    uint8_t  card  = 0;
    uint8_t  aux   = 0;
};

uint64_t    make_tag(kind k, uint8_t group, uint32_t seq, uint8_t card, uint8_t aux = 0);
tag_fields  split_tag(uint64_t tag);
std::string tag_str(uint64_t tag);  // "X-BOUNDARY group 1 seq 42 card 0 aux 3"

// X-HELLO's payload: both sides' identities, compared field by field
struct hello {
    uint32_t magic      = 0x54545331u;  // "TTS1"
    uint32_t version    = 1;
    char     commit[48] = {};           // the source commit (git rev-parse HEAD)
    char     digest[40] = {};           // the device-code digest (md5 of cuobjdump -sass of the binaries)
    char     model[72]  = {};  // the model files' and the shared settings' hashes (fn/runtime/stage.h hello_files)
    int32_t  split      = 0;   // the split layer k
    int32_t  slots      = 0;   // the slot count
    int32_t  role       = 0;   // 0 leader, 1 follower (the peer's must differ)
    int32_t  pad        = 0;
};

// the first differing field's name and both values, empty when they match (role must differ)
std::string hello_mismatch(const hello & mine, const hello & theirs);
void        set_text(char * dst, size_t cap, const std::string & s);

struct options {
    bool        leader = true;
    std::string addr;              // the follower's listen address (an IPv4 address)
    uint16_t    port      = 13371;
    double      timeout_s = 10.0;  // every operation's wait bound (the default for wait)
    double      connect_s = 60.0;  // the leader retries its connect this long; the follower waits this long for it
    hello       me;
};

// totals since start: messages and bytes each way, and the runtime's time inside wait
struct counters {
    uint64_t sent_msgs = 0, sent_bytes = 0, recv_msgs = 0, recv_bytes = 0;
    double   wait_s = 0;
};

class link {
  public:
    link();
    ~link();
    link(const link &)             = delete;
    link & operator=(const link &) = delete;

    // the context, worker and link thread; the follower's listener or the leader's connect; X-HELLO both ways.
    // false with the reason (a mismatch names the field)
    bool          start(const options & o, std::string & err);
    // the peer's X-HELLO (valid after start)
    const hello & peer() const;
    // register pinned host memory [p, p + n) once (ucp_mem_map); sends and receives inside it skip registration
    bool          map(void * p, size_t n, std::string & err);

    // post a send or receive of n bytes at p (inside a mapped range, or not); returns the operation's id (>= 0), or
    // -1 with err (the link is lost, or no operation slot is free). A receive matches tag exactly and must receive
    // exactly n bytes. Single producer: one runtime thread posts, waits and tests
    int32_t send(uint64_t tag, const void * p, size_t n, std::string & err);
    int32_t recv(uint64_t tag, void * p, size_t n, std::string & err);
    // wait for op to complete, at most timeout_s (< 0: options::timeout_s); true when it completed with its bytes.
    // On false the message names the tag; a timed-out receive is cancelled (its buffer may still be written until the
    // cancel completes, which the link thread finishes before the slot is reused). The op is done either way
    bool    wait(int32_t op, std::string & err, double timeout_s = -1);
    // without waiting: done set when the op completed (then as wait: true for its bytes, false with err)
    bool    test(int32_t op, bool & done, std::string & err);
    // wait for one of ops[0 .. n - 1] (n <= 8) to complete; which: its index. As wait otherwise
    bool    wait_any(const int32_t * ops, int32_t n, int32_t & which, std::string & err, double timeout_s = -1);

    // the endpoint's transports as UCX chose them ("tcp/enp3s0np0 ...", "rc_mlx5/mlx5_0:1 ..."), set after start
    const std::string & transport() const;
    // the peer was lost (its process ended or the connection failed); why names it
    bool                lost() const;
    std::string         lost_why() const;
    // an unexpected message waiting on the worker, if any (its tag), for a refusal's message; false: none seen
    bool                unexpected(uint64_t & tag, size_t & bytes);
    counters            totals() const;
    // close the endpoint (flush, then close) and stop the thread; the destructor does it too
    void                stop();

    struct impl;

  private:
    std::unique_ptr<impl> p_;
};

}  // namespace fn::xchg::link
