#include "fn/xchg/link_test.h"

#include "fn/xchg/link.h"

#include <cuda_runtime_api.h>
#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace fn::xchg::link {

namespace {

using clk = std::chrono::steady_clock;

// Flash-Next's crossing sizes (the stage's, fn/runtime/stage.h): a decode boundary row on one card is hres [4][1280],
// hpost [4] and hx [1280] F32; a prompt row is the replicated streams [4][2560] F32
constexpr size_t  dec_row     = (4 * 1280 + 4 + 1280) * 4;  // 25,616
constexpr size_t  prm_row     = 4 * 2560 * 4;               // 40,960
constexpr int32_t n_slot      = 8;                          // slot buffers (the split's slots)
constexpr int32_t lane_rows   = 5;                          // a lane's rows at most (R = d + 1)
constexpr int32_t piece_rows  = 1024;
constexpr int32_t n_piece     = 4;
constexpr size_t  small_bytes = 512;
constexpr size_t  step_bytes = 256, ctrl_bytes = 128, ack_bytes = 64;

double since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

uint64_t seed_of(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e) {
    return mix(mix(mix(mix(mix(a) ^ b) ^ c) ^ d) ^ e);
}

// word i of a payload: the canonical quiet NaN, both infinities, denormals, -0 and the smallest normal now and then;
// otherwise finite bits
uint32_t word(uint64_t seed, uint64_t i) {
    const uint64_t x = mix(seed ^ (i * 0xd1b54a32d192ed03ull));
    switch (x % 61) {
        case 0:
            return 0x7fc00000u;
        case 1:
            return 0x7f800000u;
        case 2:
            return 0xff800000u;
        case 3:
            return 0x00000001u;
        case 4:
            return 0x807fffffu;
        case 5:
            return 0x80000000u;
        case 6:
            return 0x00800000u;
        default:
            break;
    }
    uint32_t b = uint32_t(x >> 17);
    if ((b & 0x7f800000u) == 0x7f800000u) {
        b &= 0xbfffffffu;  // finite
    }
    return b;
}

void fill(uint8_t * p, size_t n, uint64_t seed) {
    uint32_t * w = reinterpret_cast<uint32_t *>(p);
    for (size_t i = 0; i < n / 4; ++i) {
        w[i] = word(seed, i);
    }
}

int64_t mismatches(const uint8_t * p, size_t n, uint64_t seed) {
    const uint32_t * w   = reinterpret_cast<const uint32_t *>(p);
    int64_t          bad = 0;
    for (size_t i = 0; i < n / 4; ++i) {
        bad += w[i] != word(seed, i) ? 1 : 0;
    }
    return bad;
}

struct ack_msg {
    uint32_t magic      = 0x41434b31u;
    uint32_t case_id    = 0;
    int64_t  mismatches = 0;
    int64_t  messages   = 0;
    int64_t  refused    = 0;
    int64_t  extra      = 0;
    int64_t  pad[3]     = { 0, 0, 0 };
};

static_assert(sizeof(ack_msg) == ack_bytes, "an X-ACK of the test is 64 bytes");

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e != cudaSuccess) {
        err = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

struct bufs {
    uint8_t * dec        = nullptr;  // [slot][card][5 rows] pinned
    uint8_t * prm        = nullptr;  // [piece][1024 rows] pinned
    uint8_t * small      = nullptr;  // [3][8][512] pinned: step, ctrl, ack
    uint8_t * check      = nullptr;  // a piece's read-back, pinned
    uint8_t * dev_dec[2] = { nullptr, nullptr };
    uint8_t * dev_prm[2] = { nullptr, nullptr };

    uint8_t * lane(int32_t slot, int32_t card) const {
        return dec + (size_t(slot) * 2 + size_t(card)) * lane_rows * dec_row;
    }

    uint8_t * piece(int32_t q) const { return prm + size_t(q % n_piece) * piece_rows * prm_row; }

    uint8_t * sm(int32_t kind_ix, int32_t i) const {
        return small + (size_t(kind_ix) * 8 + size_t(i % 8)) * small_bytes;
    }
};

bool alloc(bufs & b, const test_options & o, link & lk, std::string & err) {
    const size_t dec_n = size_t(n_slot) * 2 * lane_rows * dec_row;
    const size_t prm_n = size_t(n_piece) * piece_rows * prm_row;
    const size_t sm_n  = 3 * 8 * small_bytes;
    if (!cuda_ok(cudaSetDevice(o.device[0]), "cudaSetDevice", err) ||
        !cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&b.dec), dec_n, cudaHostAllocPortable), "pinned", err) ||
        !cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&b.prm), prm_n, cudaHostAllocPortable), "pinned", err) ||
        !cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&b.small), sm_n, cudaHostAllocPortable), "pinned", err) ||
        !cuda_ok(
            cudaHostAlloc(reinterpret_cast<void **>(&b.check), size_t(piece_rows) * prm_row, cudaHostAllocPortable),
            "pinned", err)) {
        return false;
    }
    if (!o.leader) {
        for (int32_t c = 0; c < 2; ++c) {
            if (!cuda_ok(cudaSetDevice(o.device[c]), "cudaSetDevice", err) ||
                !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&b.dev_dec[c]), dec_n), "cudaMalloc", err) ||
                !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&b.dev_prm[c]), size_t(piece_rows) * prm_row),
                         "cudaMalloc", err)) {
                return false;
            }
        }
    }
    return lk.map(b.dec, dec_n, err) && lk.map(b.prm, prm_n, err) && lk.map(b.small, sm_n, err);
}

void release(bufs & b) {
    for (uint8_t * p : { b.dec, b.prm, b.small, b.check }) {
        if (p != nullptr) {
            cudaFreeHost(p);
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        for (uint8_t * p : { b.dev_dec[c], b.dev_prm[c] }) {
            if (p != nullptr) {
                cudaFree(p);
            }
        }
    }
}

// the far card's check: host buffer -> card -> read back, then the bytes against the pattern
int64_t card_check(const bufs &         b,
                   const test_options & o,
                   int32_t              card,
                   uint8_t *            dev,
                   const uint8_t *      src,
                   size_t               n,
                   uint64_t             seed,
                   std::string &        err) {
    if (!cuda_ok(cudaSetDevice(o.device[card]), "cudaSetDevice", err) ||
        !cuda_ok(cudaMemcpy(dev, src, n, cudaMemcpyHostToDevice), "H2D", err) ||
        !cuda_ok(cudaMemcpy(b.check, dev, n, cudaMemcpyDeviceToHost), "D2H", err)) {
        return -1;
    }
    return mismatches(b.check, n, seed);
}

struct runner {
    const test_options & o;
    link &               lk;
    bufs &               b;
    uint32_t             seq    = 1;
    int32_t              failed = 0;
    std::string          err;

    bool on(const char * name) const {
        if (o.cases == "all") {
            return true;
        }
        const std::string s = "," + o.cases + ",";
        return s.find("," + std::string(name) + ",") != std::string::npos;
    }

    void report(const char * name, bool ok, const std::string & what) {
        std::printf("link-test %s: %s: %s\n", o.leader ? "leader" : "follower", name, ok ? "PASS" : "FAIL");
        if (!what.empty()) {
            std::printf("  %s\n", what.c_str());
        }
        std::fflush(stdout);
        failed += ok ? 0 : 1;
    }

    // the follower's verdict of a case, back to the leader (and the leader's wait for it)
    bool send_ack(uint32_t case_id, const ack_msg & a) {
        uint8_t * p = b.sm(2, int32_t(case_id));
        std::memcpy(p, &a, sizeof a);
        const int32_t op = lk.send(make_tag(kind::ack, 0, case_id, 0), p, ack_bytes, err);
        return op >= 0 && lk.wait(op, err);
    }

    bool recv_ack(uint32_t case_id, ack_msg & a, double timeout_s = -1) {
        uint8_t *     p  = b.sm(2, int32_t(case_id));
        const int32_t op = lk.recv(make_tag(kind::ack, 0, case_id, 0), p, ack_bytes, err);
        if (op < 0 || !lk.wait(op, err, timeout_s)) {
            return false;
        }
        std::memcpy(&a, p, sizeof a);
        return a.magic == 0x41434b31u && a.case_id == case_id;
    }

    // ---- decode: 1 .. 20 rows, every slot buffer, both groups and cards ----
    void decode() {
        const uint32_t cid = 1;
        int64_t        bad = 0, msgs = 0;
        bool           ok = true;
        for (int32_t g = 0; g < 2 && ok; ++g) {
            for (int32_t R = 1; R <= 20 && ok; ++R) {
                const int32_t L = (R + lane_rows - 1) / lane_rows;
                for (int32_t start = 0; start < n_slot && ok; ++start) {
                    const uint32_t s = seq++;
                    int32_t        ops[8];
                    int32_t        n = 0;
                    for (int32_t j = 0; j < L; ++j) {
                        const int32_t slot = (start + j) % n_slot;
                        const int32_t rows = j + 1 < L ? lane_rows : R - lane_rows * (L - 1);
                        for (int32_t c = 0; c < 2; ++c) {
                            const uint64_t sd  = seed_of(g, R, start, j * 2 + c, s);
                            const uint64_t tag = make_tag(kind::boundary, uint8_t(g), s, uint8_t(c), uint8_t(slot));
                            uint8_t *      p   = b.lane(slot, c);
                            if (o.leader) {
                                fill(p, rows * dec_row, sd);
                                ops[n++] = lk.send(tag, p, rows * dec_row, err);
                            } else {
                                ops[n++] = lk.recv(tag, p, rows * dec_row, err);
                            }
                            if (ops[n - 1] < 0) {
                                ok = false;
                            }
                        }
                    }
                    for (int32_t k = 0; k < n && ok; ++k) {
                        ok = lk.wait(ops[k], err);
                    }
                    if (!ok || o.leader) {
                        continue;
                    }
                    for (int32_t j = 0; j < L; ++j) {
                        const int32_t slot = (start + j) % n_slot;
                        const int32_t rows = j + 1 < L ? lane_rows : R - lane_rows * (L - 1);
                        for (int32_t c = 0; c < 2; ++c) {
                            const uint64_t sd  = seed_of(g, R, start, j * 2 + c, s);
                            uint8_t *      dev = b.dev_dec[c] + (size_t(slot) * 2 + size_t(c)) * lane_rows * dec_row;
                            const int64_t  m   = card_check(b, o, c, dev, b.lane(slot, c), rows * dec_row, sd, err);
                            if (m < 0) {
                                ok = false;
                                break;
                            }
                            bad += m;
                            ++msgs;
                        }
                    }
                }
            }
        }
        verdict("decode", cid, ok, bad, msgs, 0, 0);
    }

    // the case's verdict: the follower sends its counts, the leader waits for them and both print
    void verdict(const char * name,
                 uint32_t     cid,
                 bool         ok,
                 int64_t      bad,
                 int64_t      msgs,
                 int64_t      refused,
                 int64_t      want_refused) {
        if (!o.leader) {
            ack_msg a;
            a.case_id        = cid;
            a.mismatches     = ok ? bad : -1;
            a.messages       = msgs;
            a.refused        = refused;
            std::string what = err;
            if (!send_ack(cid, a)) {
                ok   = false;
                what = err;
            }
            report(name, ok && bad == 0 && refused == want_refused,
                   std::to_string(msgs) + " messages compared on the cards, " + std::to_string(bad) +
                       " mismatched words; refused " + std::to_string(refused) + (what.empty() ? "" : "; " + what));
            return;
        }
        ack_msg    a;
        const bool got  = ok && recv_ack(cid, a);
        const bool pass = got && a.mismatches == 0 && a.refused == want_refused;
        report(name, pass,
               got ? ("the follower compared " + std::to_string(a.messages) + " messages: " +
                      std::to_string(a.mismatches) + " mismatched words; refused " + std::to_string(a.refused)) :
                     err);
    }

    // ---- prompt: chunks of 1 .. 8,192 rows in 1,024-row pieces, four piece buffers ----
    void prompt() {
        const uint32_t cid     = 2;
        const int32_t  sizes[] = { 1, 2, 3, 511, 1023, 1024, 1025, 2047, 3000, 4096, 5121, 8191, 8192 };
        int64_t        bad = 0, msgs = 0;
        bool           ok = true;
        for (int32_t rows : sizes) {
            const uint32_t s                = seq++;
            const int32_t  P                = (rows + piece_rows - 1) / piece_rows;
            int32_t        pending[n_piece] = { -1, -1, -1, -1 };
            for (int32_t q = 0; q < P && ok; ++q) {
                const int32_t  r   = std::min(piece_rows, rows - q * piece_rows);
                const uint64_t sd  = seed_of(7, rows, q, 0, s);
                const uint64_t tag = make_tag(kind::prompt, 0, s, 0, uint8_t(q));
                uint8_t *      p   = b.piece(q);
                if (o.leader) {
                    // release: the piece buffer is refilled only after its previous send completed
                    if (pending[q % n_piece] >= 0 && !lk.wait(pending[q % n_piece], err)) {
                        ok = false;
                        break;
                    }
                    fill(p, size_t(r) * prm_row, sd);
                    pending[q % n_piece] = lk.send(tag, p, size_t(r) * prm_row, err);
                    ok                   = pending[q % n_piece] >= 0;
                } else {
                    const int32_t op = lk.recv(tag, p, size_t(r) * prm_row, err);
                    ok               = op >= 0 && lk.wait(op, err);
                    for (int32_t c = 0; c < 2 && ok; ++c) {
                        const int64_t m = card_check(b, o, c, b.dev_prm[c], p, size_t(r) * prm_row, sd, err);
                        ok              = m >= 0;
                        bad += m < 0 ? 0 : m;
                        ++msgs;
                    }
                }
            }
            for (int32_t k = 0; k < n_piece && ok && o.leader; ++k) {
                if (pending[k] >= 0) {
                    ok = lk.wait(pending[k], err);
                }
            }
        }
        verdict("prompt", cid, ok, bad, msgs, 0, 0);
    }

    // ---- small: X-STEP back to front, X-CTRL and its X-ACK both ways ----
    void small() {
        const uint32_t cid = 3;
        int64_t        bad = 0, msgs = 0;
        bool           ok = true;
        for (int32_t i = 0; i < 8 && ok; ++i) {
            const uint32_t s  = seq++;
            // X-STEP: the follower (back) sends, the leader (front) receives
            const uint64_t ts = make_tag(kind::step, uint8_t(i % 2), s, 0);
            uint8_t *      ps = b.sm(0, i);
            int32_t        op = -1;
            if (!o.leader) {
                fill(ps, step_bytes, seed_of(11, i, s, 0, 0));
                op = lk.send(ts, ps, step_bytes, err);
            } else {
                op = lk.recv(ts, ps, step_bytes, err);
            }
            ok = op >= 0 && lk.wait(op, err);
            if (ok && o.leader) {
                bad += mismatches(ps, step_bytes, seed_of(11, i, s, 0, 0));
                ++msgs;
            }
            // X-CTRL each way, each answered by an X-ACK that echoes the command's sequence
            for (int32_t dir = 0; dir < 2 && ok; ++dir) {
                const bool     sender = (dir == 0) == o.leader;
                const uint64_t tc     = make_tag(kind::ctrl, uint8_t(dir), s, 0);
                const uint64_t ta     = make_tag(kind::ack, uint8_t(dir + 4), s, 0);
                uint8_t *      pc     = b.sm(1, i);
                uint8_t *      pa     = b.sm(2, i);
                const uint64_t sd     = seed_of(13, i, s, dir, 0);
                if (sender) {
                    fill(pc, ctrl_bytes, sd);
                    const int32_t a = lk.recv(ta, pa, ack_bytes, err);
                    const int32_t c = lk.send(tc, pc, ctrl_bytes, err);
                    ok              = a >= 0 && c >= 0 && lk.wait(c, err) && lk.wait(a, err);
                    uint32_t echo   = 0;
                    std::memcpy(&echo, pa, 4);
                    bad += ok && echo != s ? 1 : 0;
                    ++msgs;
                } else {
                    const int32_t c = lk.recv(tc, pc, ctrl_bytes, err);
                    ok              = c >= 0 && lk.wait(c, err);
                    if (ok) {
                        bad += mismatches(pc, ctrl_bytes, sd);
                        ++msgs;
                        std::memset(pa, 0, ack_bytes);
                        std::memcpy(pa, &s, 4);
                        const int32_t a = lk.send(ta, pa, ack_bytes, err);
                        ok              = a >= 0 && lk.wait(a, err);
                    }
                }
            }
        }
        // both sides counted their received messages; the follower's verdict carries its own
        if (o.leader && bad != 0) {
            ok  = false;
            err = "the leader saw " + std::to_string(bad) + " mismatches";
        }
        verdict("small", cid, ok, o.leader ? 0 : bad, msgs, 0, 0);
    }

    // ---- stream: 100,000 decode messages alternating the groups ----
    void stream() {
        const uint32_t    cid    = 4;
        const int64_t     N      = o.stream_messages;
        constexpr int32_t window = 16;  // in flight: one (slot, card) buffer each
        const uint32_t    s0     = seq;
        seq += uint32_t(N);
        int64_t bad = 0, msgs = 0;
        bool    ok = true;
        int32_t ops[window];
        for (int32_t k = 0; k < window; ++k) {
            ops[k] = -1;
        }
        // message i uses buffer i % 16 (slot, card), so the window's messages never share one
        const auto msg_of = [&](int64_t i, int32_t & g, int32_t & slot, int32_t & c, int32_t & rows) {
            g    = int32_t(i % 2);
            c    = int32_t(i % 2);
            slot = int32_t((i % window) / 2);
            rows = 1 + int32_t((i * 7) % lane_rows);
        };
        // flow control: every 1,000 messages the follower acknowledges, and the leader runs at most two blocks ahead
        for (int64_t i = 0; i < N + window && ok; ++i) {
            if (i >= window) {
                // the message window slots back
                const int64_t j = i - window;
                int32_t       g, slot, c, rows;
                msg_of(j, g, slot, c, rows);
                if (!lk.wait(ops[j % window], err)) {
                    ok = false;
                    break;
                }
                if (!o.leader) {
                    bad += mismatches(b.lane(slot, c), size_t(rows) * dec_row, seed_of(17, j, s0, 0, 0));
                    ++msgs;
                    if ((j + 1) % 1000 == 0) {
                        uint8_t * p = b.sm(0, 7);
                        std::memcpy(p, &j, 8);
                        const int32_t a = lk.send(make_tag(kind::ack, 9, uint32_t(j / 1000), 0), p, 8, err);
                        ok              = a >= 0 && lk.wait(a, err);
                    }
                } else if ((j + 1) % 1000 == 0 && j >= 1999) {
                    uint8_t *     p = b.sm(0, 6);
                    const int32_t a = lk.recv(make_tag(kind::ack, 9, uint32_t((j - 1000) / 1000), 0), p, 8, err);
                    ok              = a >= 0 && lk.wait(a, err);
                }
            }
            if (i >= N || !ok) {
                continue;
            }
            int32_t g, slot, c, rows;
            msg_of(i, g, slot, c, rows);
            const uint64_t tag = make_tag(kind::boundary, uint8_t(g), s0 + uint32_t(i), uint8_t(c), uint8_t(slot));
            uint8_t *      p   = b.lane(slot, c);
            if (o.leader) {
                fill(p, size_t(rows) * dec_row, seed_of(17, i, s0, 0, 0));
                ops[i % window] = lk.send(tag, p, size_t(rows) * dec_row, err);
            } else {
                ops[i % window] = lk.recv(tag, p, size_t(rows) * dec_row, err);
            }
            ok = ops[i % window] >= 0;
        }
        // the leader's last acknowledgments
        // the loop took blocks 0 .. N / 1000 - 2 (each when the block after it was sent); the last one remains
        for (int64_t blk = N / 1000 >= 2 ? N / 1000 - 1 : 0; o.leader && ok && blk < N / 1000; ++blk) {
            uint8_t *     p = b.sm(0, 6);
            const int32_t a = lk.recv(make_tag(kind::ack, 9, uint32_t(blk), 0), p, 8, err);
            ok              = a >= 0 && lk.wait(a, err);
        }
        verdict("stream", cid, ok, bad, msgs, 0, 0);
    }

    // ---- refuse: one byte longer, one byte shorter, a tag nobody receives ----
    void refuse() {
        const uint32_t cid     = 5;
        bool           ok      = true;
        int64_t        refused = 0;
        std::string    notes;
        const uint32_t s = seq;
        seq += 4;
        const size_t   n        = 2 * dec_row;
        const uint64_t ta       = make_tag(kind::boundary, 0, s, 0, 0);
        const uint64_t tb       = make_tag(kind::boundary, 0, s + 1, 0, 0);
        const uint64_t tc       = make_tag(kind::boundary, 0, s + 2, 0, 0);
        const uint64_t tc_wrong = make_tag(kind::boundary, 0, s + 3, 0, 0);
        if (o.leader) {
            uint8_t * p = b.lane(0, 0);  // 5 rows: room for n + 4
            fill(p, n + 4, 1);
            int32_t x = lk.send(ta, p, n + 4, err);
            int32_t y = lk.send(tb, p, n - 4, err);
            int32_t z = lk.send(tc_wrong, p, n, err);
            ok        = x >= 0 && y >= 0 && z >= 0 && lk.wait(x, err) && lk.wait(y, err) && lk.wait(z, err);
            verdict("refuse", cid, ok, 0, 0, 0, 3);
            return;
        }
        uint8_t *   p = b.lane(1, 0);
        std::string e;
        int32_t     x = lk.recv(ta, p, n, err);
        if (x >= 0 && !lk.wait(x, e)) {
            ++refused;
            notes += "longer: " + e + "\n  ";
        }
        int32_t y = lk.recv(tb, p, n, err);
        if (y >= 0 && !lk.wait(y, e)) {
            ++refused;
            notes += "shorter: " + e + "\n  ";
        }
        int32_t z = lk.recv(tc, p, n, err);
        if (z >= 0 && !lk.wait(z, e, 2.0)) {
            ++refused;
            notes += "wrong tag: " + e;
        }
        // drain the wrong-tagged message
        int32_t w = lk.recv(tc_wrong, p, n, err);
        ok        = w >= 0 && lk.wait(w, err);
        std::printf("  %s\n", notes.c_str());
        verdict("refuse", cid, ok, 0, 3, refused, 3);
    }

    // ---- kill: the follower ends itself mid-transfer; the leader's waits end within their bound ----
    void kill() {
        const uint32_t s    = seq++;
        const int32_t  rows = 8192;
        const int32_t  P    = rows / piece_rows;
        if (!o.leader) {
            std::printf("link-test follower: kill: receiving two pieces, then SIGKILL (the leader judges)\n");
            std::fflush(stdout);
            for (int32_t q = 0; q < 2; ++q) {
                const int32_t op =
                    lk.recv(make_tag(kind::prompt, 1, s, 0, uint8_t(q)), b.piece(q), size_t(piece_rows) * prm_row, err);
                if (op < 0 || !lk.wait(op, err)) {
                    break;
                }
            }
            raise(SIGKILL);
            return;
        }
        const auto  t0    = clk::now();
        bool        ended = false;
        std::string what;
        int32_t     pending[n_piece] = { -1, -1, -1, -1 };
        for (int32_t q = 0; q < P && !ended; ++q) {
            uint8_t * p = b.piece(q);
            if (pending[q % n_piece] >= 0 && !lk.wait(pending[q % n_piece], what)) {
                ended = true;
                break;
            }
            fill(p, size_t(piece_rows) * prm_row, seed_of(19, q, s, 0, 0));
            pending[q % n_piece] =
                lk.send(make_tag(kind::prompt, 1, s, 0, uint8_t(q)), p, size_t(piece_rows) * prm_row, what);
            ended = pending[q % n_piece] < 0;
        }
        for (int32_t k = 0; k < n_piece && !ended; ++k) {
            if (pending[k] >= 0 && !lk.wait(pending[k], what)) {
                ended = true;
            }
        }
        if (!ended) {
            // every piece left this box (buffered by the transport): the X-ACK that never comes is the wait
            ack_msg a;
            ended = !recv_ack(99, a);
            what  = err;
        }
        const double t = since(t0);
        const bool   named =
            what.find("X-") != std::string::npos &&
            (what.find("split stage lost") != std::string::npos || what.find("no completion") != std::string::npos);
        const bool pass = ended && named && t <= o.timeout_s + 2.0;
        report("kill", pass,
               std::string(ended ? "the leader's wait ended" : "no wait ended") + " within its bound (" +
                   std::to_string(int(o.timeout_s)) + " s + 2): " + (t <= o.timeout_s + 2.0 ? "yes" : "no") +
                   "; message: " + what);
    }
};

}  // namespace

int run_test(const test_options & o) {
    link    lk;
    options lo;
    lo.leader    = o.leader;
    lo.addr      = o.addr;
    lo.port      = o.port;
    lo.timeout_s = o.timeout_s;
    lo.connect_s = 120.0;
    set_text(lo.me.commit, sizeof lo.me.commit, "link-test");
    set_text(lo.me.digest, sizeof lo.me.digest, "link-test");
    set_text(lo.me.model, sizeof lo.me.model, "none");
    lo.me.split = 0;
    lo.me.slots = n_slot;
    std::string err;
    std::printf("link-test %s: %s %s:%u, UCX_TLS=%s, UCX_NET_DEVICES=%s, UCX_IB_GID_INDEX=%s\n",
                o.leader ? "leader" : "follower", o.leader ? "connecting to" : "listening on", o.addr.c_str(),
                unsigned(o.port), getenv("UCX_TLS") ? getenv("UCX_TLS") : "(unset)",
                getenv("UCX_NET_DEVICES") ? getenv("UCX_NET_DEVICES") : "(unset)",
                getenv("UCX_IB_GID_INDEX") ? getenv("UCX_IB_GID_INDEX") : "(unset)");
    std::fflush(stdout);
    if (!lk.start(lo, err)) {
        std::printf("link-test: connect: FAIL\n  %s\n", err.c_str());
        return 1;
    }
    const bool rc = lk.transport().find("rc_mlx5") != std::string::npos;
    std::printf("link-test %s: connect: PASS (X-HELLO both ways)\nlink-test transport: %s\n",
                o.leader ? "leader" : "follower", lk.transport().c_str());
    std::printf("link-test %s: transport line rc_mlx5: %s\n", o.leader ? "leader" : "follower",
                rc ? "PASS" : "FAIL (not rc_mlx5)");
    std::fflush(stdout);
    bufs b;
    if (!alloc(b, o, lk, err)) {
        std::printf("link-test: buffers: FAIL\n  %s\n", err.c_str());
        release(b);
        return 1;
    }
    runner r{ o, lk, b, 1, 0, std::string() };
    r.failed = rc ? 0 : 1;
    if (r.on("decode")) {
        r.decode();
    }
    if (r.on("prompt")) {
        r.prompt();
    }
    if (r.on("small")) {
        r.small();
    }
    if (r.on("stream")) {
        r.stream();
    }
    if (r.on("refuse")) {
        r.refuse();
    }
    if (r.on("kill")) {
        r.kill();
    }
    const counters t = lk.totals();
    std::printf("link-test %s: link totals: sent %llu messages / %llu bytes, received %llu messages / %llu bytes\n",
                o.leader ? "leader" : "follower", (unsigned long long) t.sent_msgs, (unsigned long long) t.sent_bytes,
                (unsigned long long) t.recv_msgs, (unsigned long long) t.recv_bytes);
    std::printf("link-test %s: %s (%d failed)\n", o.leader ? "leader" : "follower", r.failed == 0 ? "PASS" : "FAIL",
                r.failed);
    std::fflush(stdout);
    lk.stop();
    release(b);
    return r.failed == 0 ? 0 : 1;
}

}  // namespace fn::xchg::link
