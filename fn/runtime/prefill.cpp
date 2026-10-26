#include "fn/runtime/prefill.h"

#include "fn/kernels/ple/ple.h"
#include "fn/model/pass.h"
#include "fn/model/state.h"

#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace fn::runtime {

namespace {

constexpr int32_t E = ple_input; // 2,560 PLE input values a row
constexpr int64_t gdn_bytes = int64_t(model::gdn_heads_card) * model::gdn_head_dim * model::gdn_head_dim * 4;
// ME11: a checkpoint's GDN layer holds one bank and one record set's capacity (5 steps)
constexpr int64_t rec_set_bytes = model::gdn_record_bytes / 2;
constexpr int64_t rec_step_bytes = model::gdn_record_step_bytes;
constexpr int64_t conv_bytes = int64_t(model::gdn_conv_channels_card) * model::gdn_conv_history * 4;
constexpr int64_t ple_hist_bytes = fn::ple::hist_version_floats * 4;
constexpr int64_t carry_bytes = int64_t(4) * 1280 * 4;
static_assert(pf_max_checkpoints == model::prefill_checkpoints, "the checkpoint class holds pf_max_checkpoints");
static_assert(pf_chunk_rows == model::prefill_stage_rows, "a staging slot holds one chunk");
static_assert(pf_chunk_rows == 2 * pf_half_rows && pf_chunk_rows <= pf_span, "FC-PF3's chunk and halves");
static_assert(pf_split_rows <= pf_half_rows && pf_cached_split_rows <= pf_half_rows && pf_half_rows % 64 == 0,
              "a split chunk's halves");
static_assert(pf_half_rows % pf_slice_rows == 0 && pf_span % pf_slice_rows == 0, "slices tile the halves");

using clk = std::chrono::steady_clock;

bool fail(std::string & err, const std::string & what) {
    err = what;
    return false;
}

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e != cudaSuccess) {
        err = std::string("prefill: ") + what + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

double since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

} // namespace

uint64_t token_key(const int32_t * tok, int64_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t v = static_cast<uint32_t>(tok[i]);
        for (int32_t b = 0; b < 4; ++b) {
            h = (h ^ ((v >> (8 * b)) & 0xffu)) * 0x100000001b3ull;
        }
    }
    return h;
}

int32_t checkpoint_book::slot_for(int64_t pos, int32_t owner) const {
    int32_t empty = -1, oldest = 0;
    for (int32_t i = 0; i < pf_max_checkpoints; ++i) {
        if (e_[i].pos == pos && e_[i].owner == owner) {
            return i;
        }
        if (e_[i].pos < 0 && empty < 0) {
            empty = i;
        }
        if (e_[i].age < e_[oldest].age) {
            oldest = i;
        }
    }
    return empty >= 0 ? empty : oldest;
}

void checkpoint_book::record(int32_t slot, int64_t pos, uint64_t key, int32_t owner) {
    e_[slot].pos = pos;
    e_[slot].key = key;
    e_[slot].age = next_age_++;
    e_[slot].owner = owner;
}

int32_t checkpoint_book::latest(int64_t n, int32_t owner) const {
    int32_t best = -1;
    for (int32_t i = 0; i < pf_max_checkpoints; ++i) {
        if (e_[i].pos >= 0 && e_[i].owner == owner && e_[i].pos <= n && (best < 0 || e_[i].pos > e_[best].pos)) {
            best = i;
        }
    }
    return best;
}

void checkpoint_book::drop(int32_t slot) {
    e_[slot] = pf_checkpoint();
}

void checkpoint_book::drop_after(int64_t n, int32_t owner) {
    for (pf_checkpoint & c : e_) {
        if (c.pos > n && c.owner == owner) {
            c = pf_checkpoint();
        }
    }
}

void checkpoint_book::clear() {
    for (pf_checkpoint & c : e_) {
        c = pf_checkpoint();
    }
}

void checkpoint_book::clear_owner(int32_t owner) {
    for (pf_checkpoint & c : e_) {
        if (c.owner == owner) {
            c = pf_checkpoint();
        }
    }
}

int32_t checkpoint_book::count() const {
    int32_t n = 0;
    for (const pf_checkpoint & c : e_) {
        n += c.pos >= 0 ? 1 : 0;
    }
    return n;
}

int32_t checkpoint_book::count_owner(int32_t owner) const {
    int32_t n = 0;
    for (const pf_checkpoint & c : e_) {
        n += c.pos >= 0 && c.owner == owner ? 1 : 0;
    }
    return n;
}

int32_t choose_checkpoint(checkpoint_book & book, const std::vector<int32_t> & all, int64_t n_reuse, int32_t owner) {
    // at least the last token is processed, so its logits exist (FC-PF1's rule for nothing new)
    const int64_t keep = std::min<int64_t>(n_reuse, static_cast<int64_t>(all.size()) - 1);
    int32_t slot = -1;
    while (keep > 0 && (slot = book.latest(keep, owner)) >= 0) {
        if (book.at(slot).key == token_key(all.data(), book.at(slot).pos)) {
            return slot;
        }
        book.drop(slot); // its prefix is not the prompt's
    }
    return -1;
}

prefiller::~prefiller() {
    join();
    for (int32_t c = 0; c < 2 && m_ != nullptr; ++c) {
        cudaSetDevice(m_->device[c]);
        for (int32_t s = 0; s < 2; ++s) {
            if (uploaded_[s][c] != nullptr) {
                cudaEventDestroy(static_cast<cudaEvent_t>(uploaded_[s][c]));
            }
            if (read_[s][c] != nullptr) {
                cudaEventDestroy(static_cast<cudaEvent_t>(read_[s][c]));
            }
            if (half_[s][c] != nullptr) {
                cudaEventDestroy(static_cast<cudaEvent_t>(half_[s][c]));
            }
        }
        if (copy_[c] != nullptr) {
            cudaStreamSynchronize(static_cast<cudaStream_t>(copy_[c]));
            cudaStreamDestroy(static_cast<cudaStream_t>(copy_[c]));
        }
    }
    for (float *& h : host_) {
        if (h != nullptr) {
            cudaFreeHost(h);
            h = nullptr;
        }
    }
    if (lane_host_ != nullptr) {
        cudaFreeHost(lane_host_);
        lane_host_ = nullptr;
    }
}

bool prefiller::init(target & t, model::model & m, std::string & err) {
    m_ = &m;
    const model::reserved_class * ck = m.state_class("prefill checkpoints (PF1)");
    const model::reserved_class * st = m.state_class("prefill staging (PF1)");
    if (ck == nullptr || st == nullptr || (int32_t)ck->plan.parts.size() != pf_max_checkpoints ||
        st->plan.parts.size() != 2) {
        return fail(err, "prefill: the checkpoint and staging classes (PF1) are not reserved");
    }
    model::state_params sp;
    sp.ple_conv_kernel = m.ple_conv_kernel;
    sp.ple_ngram = m.ple_ngram;
    sp.win    = m.win;  // TTTS1: the box's GDN layers, its PLE history (layer 1) and the draft's carry (the head)
    ck_bytes_ = model::checkpoint_bytes(sp);
    if (ck_bytes_ != m.win.gdn_layers() * (gdn_bytes + rec_set_bytes + conv_bytes) +
                         (m.win.ple() ? ple_hist_bytes : 0) + (m.win.head ? carry_bytes : 0)) {
        return fail(err, "prefill: a checkpoint's size is not the PLE history's and the GDN states' (FC-ST1)");
    }
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t i = 0; i < pf_max_checkpoints; ++i) {
            ck_[i][c] = ck->part_ptr(i, c);
            if (ck_[i][c] == nullptr || ck->plan.parts[i].bytes[c] != ck_bytes_) {
                return fail(err, "prefill: checkpoint " + std::to_string(i) + " is not reserved on card " +
                                     std::to_string(c));
            }
        }
        for (int32_t s = 0; s < 2; ++s) {
            if (m.single && c == 1) {
                dev_[s][1] = dev_[s][0];
                continue;
            }
            // the staging part is pf_chunk_rows x 2,560 F32, 256-byte aligned (state.cpp)
            dev_[s][c] = reinterpret_cast<float *>(st->part_ptr(s, c));
            if (dev_[s][c] == nullptr || st->plan.parts[s].bytes[c] < int64_t(pf_chunk_rows) * E * 4) {
                return fail(err, "prefill: staging slot " + std::to_string(s) + " is not reserved");
            }
        }
        cudaStream_t s = nullptr;
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device", err) ||
            !cuda_ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "copy stream", err)) {
            return false;
        }
        copy_[c] = s;
        for (int32_t k = 0; k < 2; ++k) {
            cudaEvent_t a = nullptr, b = nullptr, h = nullptr;
            if (!cuda_ok(cudaEventCreateWithFlags(&a, cudaEventDisableTiming), "event", err) ||
                !cuda_ok(cudaEventCreateWithFlags(&b, cudaEventDisableTiming), "event", err) ||
                !cuda_ok(cudaEventCreateWithFlags(&h, cudaEventDisableTiming), "event", err)) {
                return false;
            }
            uploaded_[k][c] = a;
            read_[k][c] = b;
            half_[k][c] = h;
        }
    }
    for (float *& h : host_) {
        void * p = nullptr;
        if (!cuda_ok(cudaHostAlloc(&p, size_t(pf_chunk_rows) * E * 4, cudaHostAllocPortable), "pinned staging", err)) {
            return false;
        }
        h = static_cast<float *>(p);
    }
    // PB1: the bulk path (its arena, the replicas filled card to card, its copy streams and events)
    pp_                       = std::make_unique<model::prompt_path>();
    model::pass_card *  pc[2] = { &t.card(0), &t.card(1) };
    model::draft_card * dc[2] = { t.draft_on() ? &t.draft(0) : nullptr, t.draft_on() ? &t.draft(1) : nullptr };
    const model::expert_service * x = t.card(0).c.ex_service;
    if (x != nullptr && x->helper_cpus() != nullptr) {
        helper_cpus_ = *x->helper_cpus();
    }
    return pp_->init(m, pc, dc, err);
}

bool prefiller::set_options(int32_t short_read, int32_t chunk_rows, std::string & err) {
    if (short_read < 0 || short_read > pf_chunk_rows || chunk_rows < 256 || chunk_rows > pf_chunk_rows ||
        chunk_rows % 256 != 0) {
        return fail(err, "prefill: a short read outside 0 .. 8,192 or a chunk outside 256 .. 8,192 in 256s");
    }
    short_read_ = short_read;
    chunk_rows_ = chunk_rows;
    return true;
}

void prefiller::loan_spans(const std::vector<pf_chunk> & plan, std::vector<model::byte_span> & out) const {
    int32_t rows[2] = {0, 0};
    for (size_t k = 0; k < plan.size(); ++k) {
        rows[k % 2] = std::max(rows[k % 2], plan[k].rows);
    }
    for (int32_t sl = 0; sl < 2; ++sl) {
        for (int32_t c = 0; c < (m_->single ? 1 : 2) && rows[sl] > 0; ++c) {
            out.push_back({dev_[sl][c], int64_t(rows[sl]) * E * 4});
        }
    }
    if (bulk_) {
        pp_->arena_spans(out);
    }
}

bool prefiller::loan_check(std::string & err) {
    const model::expert_service * x = pp_ != nullptr ? pp_->service() : nullptr;
    if (x == nullptr || !x->prompt_lent()) {
        const std::vector<model::byte_span> none;
        return x == nullptr || pp_->loan_check(none, err);
    }
    // the largest chunk in both staging slots, the arena laid out for it (MX3: the cached one-half bound)
    std::vector<pf_chunk> two(2);
    two[0].rows = two[1].rows = chunk_rows_;
    if (!pp_->layout(chunk_rows_, pf_max_half(chunk_rows_, legacy_ ? pf_split_rows : pf_cached_split_rows), err)) {
        return false;
    }
    std::vector<model::byte_span> spans;
    loan_spans(two, spans);
    return pp_->loan_check(spans, err);
}

// ME3: a known-token short read: forced target rounds of at most five rows (the windows, through the expert service
// like decode, no prompt buffer and no loan), each followed by the draft's catch-up, stopping at every multiple of
// 8,192 for its checkpoint as the bulk path does; the handoff state as the forced path's
bool prefiller::run_short(target & t, pf_run_stats & s, std::string & err) {
    model::expert_service * x = t.card(0).c.ex_service;
    struct marker {
        model::expert_service * x;
        ~marker() { x->set_short_read(false); }
    } mk{x};
    x->set_short_read(true);
    round_result r;
    const int64_t to = s.to;
    while (t.slot().u > 0) {
        const int32_t B = pf_short_rows(t.slot().n, t.slot().u);
        if (!t.forced(B, r, err)) {
            return false;
        }
        ++s.rounds;
        if (t.draft_on()) {
            if (!t.draft_catchup(B, r, err)) {
                return false;
            }
            ++s.catchups;
        }
        const int64_t pos = t.slot().n;
        if (checkpoint_due(pos, to)) {
            const int32_t slot = book_.slot_for(pos, t.slot_index());
            if (!save(t, slot, err)) {
                book_.drop(slot);
                return false;
            }
            book_.record(slot, pos, token_key(tok_.data(), pos), t.slot_index());
            ++s.checkpoints;
        }
    }
    const ledger & e = t.slot();
    if (e.n != to || e.u != 0 || !e.retained || e.o != 0 || (t.draft_on() && t.draft_n() != to)) {
        return fail(err, "prefill: the short read's handoff state is not N = L, U = 0, retained, O = 0 and the draft "
                         "at N");
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "checkpoints", err)) {
            return false;
        }
    }
    return true;
}

bool prefiller::gather_rows(const ple_params & p, const uint8_t * table, const std::vector<int32_t> & tok,
                            int64_t start, int32_t rows, float * e, std::string & err) {
    if (table == nullptr || e == nullptr || start < 0 || rows < 1 || start + rows > static_cast<int64_t>(tok.size())) {
        return fail(err, "prefill gather: rows outside the tokens");
    }
    // every group's pages are hinted before the first gather (the reads queue together); then the groups are
    // gathered. Each group is a decode round's call: the rows are the decode path's bitwise (the indices take a row's
    // predecessors from the tokens, whichever group they fall in)
    std::vector<int32_t> idx(size_t(rows) * ple_heads);
    for (int32_t g = 0; g < rows; g += pf_slice_rows) {
        const int32_t n = std::min(pf_slice_rows, rows - g);
        const int64_t pos = start + g;
        int32_t n_ranges = 0;
        if (!ple_indices(p, tok.data(), pos, tok.data() + pos, n, idx.data() + size_t(g) * ple_heads)) {
            return fail(err, "prefill gather: PLE indices at " + std::to_string(pos));
        }
        if (!ple_prefetch(p, table, idx.data() + size_t(g) * ple_heads, n, n_ranges, err)) {
            return false;
        }
    }
    for (int32_t g = 0; g < rows; g += pf_slice_rows) {
        const int32_t n = std::min(pf_slice_rows, rows - g);
        if (!ple_gather(p, table, idx.data() + size_t(g) * ple_heads, n, e + size_t(g) * E)) {
            return fail(err, "prefill gather: PLE gather at " + std::to_string(start + g));
        }
    }
    return true;
}

uint64_t pf_helper_affinity(const std::vector<int32_t> & cpus, bool apply) {
    if (!cpus.empty() && apply) {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (const int32_t cpu : cpus) {
            CPU_SET(cpu, &set);
        }
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    cpu_set_t got;
    CPU_ZERO(&got);
    uint64_t mask = 0;
    if (pthread_getaffinity_np(pthread_self(), sizeof(got), &got) == 0) {
        for (int32_t cpu = 0; cpu < 64; ++cpu) {
            mask |= CPU_ISSET(cpu, &got) ? uint64_t(1) << cpu : 0;
        }
    }
    return mask;
}

void prefiller::start_gather(const ple_params & p, int32_t s, const pf_chunk & c, const std::vector<int32_t> * tok) {
    gathered_.store(false, std::memory_order_relaxed);
    gather_ok_ = false;
    gather_err_.clear();
    gather_tok_ = tok != nullptr ? tok : &tok_;
    gather_ = std::thread([this, &p, s, c] {
        // ME3: the helpers' mask (expert_service::helper_cpus), not the decode thread's CPU this thread inherits; the
        // mask read back either way (ME3's base arm keeps the inherited one)
        gather_mask_ = pf_helper_affinity(helper_cpus_, !legacy_);
        const auto t0 = clk::now();
        gather_ok_ = gather_rows(p, m_->ple_table, *gather_tok_, c.start, c.rows, host_[s], gather_err_);
        gather_s_ = since(t0);
        gathered_.store(true, std::memory_order_release);
    });
}

void prefiller::join() {
    if (gather_.joinable()) {
        gather_.join();
    }
}

bool prefiller::upload(int32_t s, int32_t rows, std::string & err) {
    for (int32_t c = 0; c < (m_->single ? 1 : 2); ++c) {
        cudaStream_t cs = static_cast<cudaStream_t>(copy_[c]);
        // the device slot's last readers (the rounds of the chunk two back) complete before it is overwritten
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
            !cuda_ok(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(read_[s][c]), 0), "wait readers", err) ||
            !cuda_ok(cudaMemcpyAsync(dev_[s][c], host_[s], size_t(rows) * E * 4, cudaMemcpyHostToDevice, cs),
                     "staging upload", err) ||
            !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(uploaded_[s][c]), cs), "record upload", err)) {
            return false;
        }
    }
    return true;
}

bool prefiller::save(target & t, int32_t slot, std::string & err) {
    const uint32_t cur = t.slot().cur; // version 0 is slot cur (SR1)
    // ME11: the committed GDN matrix is the selector's bank with its k live records; both are copied with k
    const uint32_t gsel = t.slot().gdn;
    const int64_t bank = fn::round::gdn_bank(gsel), set = fn::round::gdn_set(gsel);
    const int32_t steps = fn::round::gdn_steps(gsel);
    ck_gdn_steps_[slot] = steps;
    for (int32_t c = 0; c < 2; ++c) {
        model::pass_card & p = t.card(c);
        cudaStream_t s = static_cast<cudaStream_t>(t.stream(c));
        uint8_t * q = ck_[slot][c];
        int64_t off = 0;
        int32_t gdn = 0;
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err)) {
            return false;
        }
        for (const model::layer_state & st : p.s) {
            if (st.gdn == nullptr) {
                continue;
            }
            ++gdn;
            if (!cuda_ok(cudaMemcpyAsync(q + off, reinterpret_cast<const uint8_t *>(st.gdn) + bank * gdn_bytes,
                                         size_t(gdn_bytes), cudaMemcpyDefault, s),
                         "checkpoint GDN state", err) ||
                (steps > 0 &&
                 !cuda_ok(cudaMemcpyAsync(q + off + gdn_bytes,
                                          reinterpret_cast<const uint8_t *>(st.rec) + set * rec_set_bytes,
                                          size_t(steps * rec_step_bytes), cudaMemcpyDefault, s),
                          "checkpoint GDN records", err)) ||
                !cuda_ok(cudaMemcpyAsync(q + off + gdn_bytes + rec_set_bytes,
                                         reinterpret_cast<const uint8_t *>(st.conv) + cur * conv_bytes,
                                         size_t(conv_bytes), cudaMemcpyDefault, s),
                         "checkpoint GDN history", err)) {
                return false;
            }
            off += gdn_bytes + rec_set_bytes + conv_bytes;
        }
        if (gdn != m_->win.gdn_layers() ||
            (p.ple_hist != nullptr &&
             !cuda_ok(cudaMemcpyAsync(q + off, reinterpret_cast<const uint8_t *>(p.ple_hist) + cur * ple_hist_bytes,
                                      size_t(ple_hist_bytes), cudaMemcpyDefault, s),
                      "checkpoint PLE history", err))) {
            return err.empty() ? fail(err, "prefill: not the window's GDN layers") : false;
        }
        off += p.ple_hist != nullptr ? ple_hist_bytes : 0;  // TTTS1: the PLE history is the front's
        if (t.draft_on() &&
            !cuda_ok(cudaMemcpyAsync(q + off, t.draft(c).carry_prev, size_t(carry_bytes), cudaMemcpyDefault, s),
                     "checkpoint retained carry", err)) {
            return false;
        }
    }
    return true;
}

bool prefiller::load(target & t, int32_t slot, std::string & err) {
    const uint32_t cur = t.slot().cur; // 0 after rewind
    const int32_t steps = ck_gdn_steps_[slot]; // ME11: into bank 0 and set 0 (target::restored sets the selector)
    for (int32_t c = 0; c < 2; ++c) {
        model::pass_card & p = t.card(c);
        cudaStream_t s = static_cast<cudaStream_t>(t.stream(c));
        const uint8_t * q = ck_[slot][c];
        int64_t off = 0;
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err)) {
            return false;
        }
        for (const model::layer_state & st : p.s) {
            if (st.gdn == nullptr) {
                continue;
            }
            if (!cuda_ok(cudaMemcpyAsync(st.gdn, q + off, size_t(gdn_bytes), cudaMemcpyDefault, s),
                         "restore GDN state", err) ||
                (steps > 0 && !cuda_ok(cudaMemcpyAsync(st.rec, q + off + gdn_bytes, size_t(steps * rec_step_bytes),
                                                       cudaMemcpyDefault, s),
                                       "restore GDN records", err)) ||
                !cuda_ok(cudaMemcpyAsync(reinterpret_cast<uint8_t *>(st.conv) + cur * conv_bytes,
                                         q + off + gdn_bytes + rec_set_bytes, size_t(conv_bytes), cudaMemcpyDefault, s),
                         "restore GDN history", err)) {
                return false;
            }
            off += gdn_bytes + rec_set_bytes + conv_bytes;
        }
        if (p.ple_hist != nullptr &&
            !cuda_ok(cudaMemcpyAsync(reinterpret_cast<uint8_t *>(p.ple_hist) + cur * ple_hist_bytes, q + off,
                                     size_t(ple_hist_bytes), cudaMemcpyDefault, s),
                     "restore PLE history", err)) {
            return false;
        }
        off += p.ple_hist != nullptr ? ple_hist_bytes : 0;
        if ((t.draft_on() &&
             !cuda_ok(cudaMemcpyAsync(t.draft(c).carry_prev, q + off, size_t(carry_bytes), cudaMemcpyDefault, s),
                      "restore retained carry", err)) ||
            !cuda_ok(cudaStreamSynchronize(s), "restore", err)) {
            return false;
        }
    }
    return true;
}

bool prefiller::checkpoint_now(target & t, std::string & err) {
    const ledger & l = t.slot();
    if (l.n < 1) {
        return fail(err, "checkpoint_now: nothing processed");
    }
    const int32_t slot = book_.slot_for(l.n, t.slot_index());
    if (!save(t, slot, err)) {
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "checkpoint", err)) {
            return false;
        }
    }
    book_.record(slot, l.n, token_key(l.tok.data(), l.n), t.slot_index());
    return true;
}

bool prefiller::begin(target & t, const std::vector<int32_t> & all, int64_t n_reuse, int64_t & from,
                      std::string & err) {
    from = 0;
    if (all.empty() || n_reuse < 0) {
        return fail(err, "prefill: no tokens");
    }
    const int32_t owner = t.slot_index(); // MU1: the slot's checkpoints only
    const int32_t slot = choose_checkpoint(book_, all, n_reuse, owner);
    if (slot < 0) {
        book_.clear_owner(owner);
        if (!t.reset(err)) {
            return false;
        }
        t.set_prompt(all);
        return true;
    }
    const int64_t c = book_.at(slot).pos;
    if (!t.rewind(err) || !load(t, slot, err) || !t.restored(c, all, ck_gdn_steps_[slot], err)) {
        book_.clear_owner(owner); // the slot's state is unknown
        return false;
    }
    book_.drop_after(c, owner); // the re-prefill rewrites the state past c
    from = c;
    return true;
}

bool prefiller::run(target & t, pf_run_stats * st, std::string & err, const prefill_progress & progress) {
    const auto t0 = clk::now();
    const ledger & l = t.slot();
    const int64_t from = l.n, to = static_cast<int64_t>(l.tok.size());
    if (from >= to || l.u != to - from) {
        return fail(err, "prefill: no unprocessed prompt rows (N " + std::to_string(from) + ", L " +
                             std::to_string(to) + ")");
    }
    if (t.draft_on() && t.draft_n() != from) {
        return fail(err, "prefill: the draft is not at N before the prompt");
    }
    if (busy_[0] >= 0) {
        return fail(err, "prefill: slot " + std::to_string(busy_[0]) + "'s chunk is in progress in the main arena");
    }
    {
        // MU1: the prompt path runs this slot's chunks (one slot: the same frame as at init)
        model::pass_card * pc[2] = {&t.card(0), &t.card(1)};
        model::draft_card * dc[2] = {t.draft_on() ? &t.draft(0) : nullptr, t.draft_on() ? &t.draft(1) : nullptr};
        pp_->bind(pc, dc);
        cont_job_[0] = -1; // the run's run_start rebuilds the arena's predecessor and history
        dev_tag_job_[0] = dev_tag_job_[1] = -1;
    }
    // ME3: a known-token short read (with the expert service, not the base arm): no staging, arena or loan. MX3: with
    // the expert service (not the base arm) a chunk of at most 4,096 rows runs as one half (the ring and its routing)
    model::expert_service * const xs = t.card(0).c.ex_service;
    const int32_t split = xs != nullptr && !legacy_ ? pf_cached_split_rows : pf_split_rows;
    const std::vector<pf_chunk> plan = plan_chunks(from, to, legacy_ ? pf_chunk_rows : chunk_rows_, split);
    tok_ = l.tok;
    pf_run_stats local;
    pf_run_stats & s = st != nullptr ? *st : local;
    s = pf_run_stats();
    s.from = from;
    s.to = to;
    s.t0             = t0;
    if (pf_short_read(from, to, short_read_, xs != nullptr, legacy_)) {
        s.short_read = true;
        const bool ok = run_short(t, s, err);
        s.wall_s      = since(t0);
        s.ready_s     = ok ? s.wall_s : -1;
        return ok;
    }
    // the staging thread is joined on every exit
    struct joiner {
        prefiller * p;
        ~joiner() { p->join(); }
    } jn{this};
    const ple_params & pp = t.ple();
    // ME2b: one card with the expert cache lends PF1's staging and PB1's arena to the pool between prompts. Borrowed
    // here, before their first write (the staging upload below, run_bulk's run_start); returned after the last reader
    // (the head input, the forced path's rounds and draft catch-up, the staging copies): on success below, on every
    // other exit by the guard after the failure path has drained the streams (prompt_hold drains the device again)
    struct holder {
        model::prompt_path * pp;
        bool on = false;
        ~holder() {
            std::string e;
            if (on && !pp->hold(false, e)) {
                std::fprintf(stderr, "prefill: the prompt region was not returned: %s\n", e.c_str());
            }
        }
    } hd{ pp_.get() };
    // ME3: the sized loan: with a lent region the arena laid out for the plan's largest chunk and half (a prefix of
    // the arena), then the spans the prompt writes (its staging slots' rows, the layout) borrowed before the first
    // writer (the staging upload below, run_bulk's run_start); the base arm borrows the whole region at the full
    // layout. Nothing lent: the zone's staging only
    std::vector<model::byte_span> spans;
    const bool lent = xs != nullptr && xs->prompt_lent();
    if (lent) {
        int32_t rc = 0, rh = 0;
        for (const pf_chunk & ch : plan) {
            rc = std::max(rc, ch.rows);
            rh = std::max({rh, ch.half[0], ch.half[1]});
        }
        if (legacy_) {
            rc = pf_chunk_rows;
            rh = pf_half_rows;
        }
        if (!pp_->layout(rc, rh, err)) {
            return false;
        }
        loan_spans(plan, spans);
    }
    s.layout_rows = pp_->layout_rows();
    s.layout_half = pp_->layout_half();
    s.loan_span_bytes = lent && legacy_ ? -1 : 0;
    for (const model::byte_span & x : spans) {
        s.loan_span_bytes += lent && !legacy_ ? x.bytes : 0;
    }
    // ME4: a run of one-half chunks stages through the ring (prompt_plan.h); one with a two-half chunk, the layer buffers.
    // MX3: the ring is sized by the run's largest chunk
    if (xs != nullptr) {
        bool one = true;
        int32_t rc = 0;
        for (const pf_chunk & ch : plan) {
            one = one && ch.half[1] == 0;
            rc = std::max(rc, ch.rows);
        }
        xs->set_prompt_ring(one && bulk_ && !legacy_ ? rc : 0);
    }
    if (!pp_->hold(true, err, lent && legacy_ ? nullptr : &spans)) {
        return false;
    }
    hd.on = true;
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t k = 0; k < 2; ++k) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                !cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(uploaded_[k][c])), "host slot", err)) {
                return false;
            }
        }
    }
    if (m_->win.ple()) {  // TTTS1: the PLE rows are the front's; a back stage stages none
        start_gather(pp, 0, plan[0]);
        join();
        if (!gather_ok_) {
            return fail(err, gather_err_);
        }
        if (!upload(0, plan[0].rows, err)) {
            return false;
        }
    }
    s.bulk        = bulk_;
    const bool ok = bulk_ ? run_bulk(t, s, plan, progress, t0, err) : run_forced(t, s, plan, progress, err);
    s.helper_mask = gather_mask_;
    if (ok) {
        hd.on = false;
        if (!pp_->hold(false, err)) {
            return false;
        }
        s.wall_s  = since(t0);
        s.ready_s = s.wall_s;
        return true;
    }
    s.wall_s = since(t0);
    // a failure: both cards drained before the caller decides the slot's fate (the joiner joins the staging thread)
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m_->device[c]);
        cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c)));
        cudaStreamSynchronize(static_cast<cudaStream_t>(copy_[c]));
    }
    return false;
}

bool prefiller::run_forced(target &                      t,
                           pf_run_stats &                s,
                           const std::vector<pf_chunk> & plan,
                           const prefill_progress &      progress,
                           std::string &                 err) {
    const int64_t      from = s.from, to = s.to;
    const ple_params & pp = t.ple();
    round_result r;
    for (size_t k = 0; k < plan.size(); ++k) {
        const pf_chunk & ch = plan[k];
        const int32_t sl = static_cast<int32_t>(k % 2);
        pf_chunk_stats cs;
        cs.c = ch;
        cs.gather_s  = gather_s_;
        bool next_up = k + 1 >= plan.size();
        if (!next_up) {
            // the next chunk's pinned slot is free once its last upload (the chunk before this one) completed
            for (int32_t c = 0; c < 2; ++c) {
                if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                    !cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(uploaded_[1 - sl][c])), "host slot", err)) {
                    return false;
                }
            }
            start_gather(pp, 1 - sl, plan[k + 1]);
        }
        const auto tc = clk::now();
        for (int32_t c = 0; c < 2; ++c) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                !cuda_ok(cudaStreamWaitEvent(static_cast<cudaStream_t>(t.stream(c)),
                                             static_cast<cudaEvent_t>(uploaded_[sl][c]), 0),
                         "wait staging", err)) {
                return false;
            }
        }
        int32_t row = 0;
        for (int32_t h = 0; h < 2; ++h) {
            const int32_t end = row + ch.half[h];
            while (row < end) {
                // FC-PF3's boundary, Stage 1: the decode pass over a slice of the half
                const int32_t B = std::min(pf_slice_rows, end - row);
                const float * e[2] = {dev_[sl][0] + size_t(row) * E, dev_[sl][1] + size_t(row) * E};
                if (!t.forced_staged(B, e, r, err)) {
                    return false;
                }
                ++s.rounds;
                if (t.draft_on()) {
                    if (!t.draft_catchup(B, r, err)) {
                        return false;
                    }
                    ++s.catchups;
                }
                row += B;
                if (!next_up && gathered_.load(std::memory_order_acquire)) {
                    join();
                    if (!gather_ok_ || !upload(1 - sl, plan[k + 1].rows, err)) {
                        return gather_ok_ ? false : fail(err, gather_err_);
                    }
                    next_up = true;
                }
            }
            for (int32_t c = 0; c < 2 && ch.half[h] > 0; ++c) {
                if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                    !cuda_ok(
                        cudaEventRecord(static_cast<cudaEvent_t>(half_[h][c]), static_cast<cudaStream_t>(t.stream(c))),
                        "record half", err)) {
                    return false;
                }
            }
        }
        for (int32_t c = 0; c < 2; ++c) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                !cuda_ok(
                    cudaEventRecord(static_cast<cudaEvent_t>(read_[sl][c]), static_cast<cudaStream_t>(t.stream(c))),
                    "record readers", err)) {
                return false;
            }
        }
        const int64_t pos = ch.start + ch.rows;
        if (t.slot().n != pos || (t.draft_on() && t.draft_n() != pos)) {
            return fail(err,
                        "prefill: chunk at " + std::to_string(ch.start) + " did not end at " + std::to_string(pos));
        }
        if (checkpoint_due(pos, to)) {
            const int32_t slot = book_.slot_for(pos, t.slot_index());
            if (!save(t, slot, err)) {
                book_.drop(slot);
                return false;
            }
            book_.record(slot, pos, token_key(tok_.data(), pos), t.slot_index());
            cs.checkpoint = slot;
            ++s.checkpoints;
        }
        cs.compute_s = since(tc);
        if (!next_up) {
            const auto tw = clk::now();
            join();
            if (!gather_ok_ || !upload(1 - sl, plan[k + 1].rows, err)) {
                return gather_ok_ ? false : fail(err, gather_err_);
            }
            cs.stall_s = since(tw); // the next chunk's rows were not staged by this chunk's end: the wait for them
        }
        s.chunks.push_back(cs);
        if (progress && k + 1 < plan.size() && !progress(from, pos, to)) {
            // stopped: the chunk's rounds and checkpoint finish before the caller forgets the slot's state
            for (int32_t c = 0; c < 2; ++c) {
                if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                    !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "stop", err)) {
                    return false;
                }
            }
            s.stopped = pos;
            return fail(err, "prefill: stopped at " + std::to_string(pos));
        }
    }
    // the handoff (FC-PF2): every prompt row processed, the last one's logits retained, the draft at N
    const ledger & e = t.slot();
    if (e.n != to || e.u != 0 || !e.retained || e.o != 0 || (t.draft_on() && t.draft_n() != to)) {
        return fail(err, "prefill: the handoff state is not N = L, U = 0, retained, O = 0 and the draft at N");
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "checkpoints", err)) {
            return false;
        }
    }
    return true;
}

bool prefiller::run_bulk(target & t, pf_run_stats & s, const std::vector<pf_chunk> & plan,
                         const prefill_progress & progress, clk::time_point t0, std::string & err) {
    const int32_t nc = m_->single ? 1 : 2;
    const int64_t from = s.from, to = s.to;
    const ple_params & pp = t.ple();
    if (!pp_->run_start(t.slot().cur, err)) {
        return false;
    }
    model::prompt_chunk last;
    for (size_t k = 0; k < plan.size(); ++k) {
        const pf_chunk & ch = plan[k];
        const int32_t    sl = static_cast<int32_t>(k % 2);
        pf_chunk_stats   cs;
        cs.c         = ch;
        cs.gather_s = gather_s_;
        const bool ple     = m_->win.ple();  // TTTS1: a back stage stages no PLE rows
        bool       next_up = k + 1 >= plan.size() || !ple;
        if (!next_up) {
            for (int32_t c = 0; c < nc; ++c) {
                if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                    !cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(uploaded_[1 - sl][c])), "host slot", err)) {
                    return false;
                }
            }
            start_gather(pp, 1 - sl, plan[k + 1]);
        }
        const auto tc = clk::now();
        for (int32_t c = 0; c < nc && ple; ++c) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                !cuda_ok(cudaStreamWaitEvent(static_cast<cudaStream_t>(t.stream(c)),
                                             static_cast<cudaEvent_t>(uploaded_[sl][c]), 0),
                         "wait staging", err)) {
                return false;
            }
        }
        model::prompt_chunk pc;
        pc.index   = static_cast<int64_t>(k);
        pc.seq     = seq_++;
        pc.start   = ch.start;
        pc.rows    = ch.rows;
        pc.half[0] = ch.half[0];
        pc.half[1] = ch.half[1];
        pc.tok     = tok_.data() + ch.start;
        pc.e[0]    = dev_[sl][0];
        pc.e[1]    = dev_[sl][1];
        pc.draft   = t.draft_on();
        pc.more    = k + 1 < plan.size();
        if (!t.prompt_begin(err) || !chunk_window(t, *pp_, pc, err)) {
            return false;
        }
        for (int32_t c = 0; c < nc; ++c) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                !cuda_ok(
                    cudaEventRecord(static_cast<cudaEvent_t>(read_[sl][c]), static_cast<cudaStream_t>(t.stream(c))),
                    "record readers", err)) {
                return false;
            }
        }
        // the chunk's end: both cards joined, then its errors, then its publication (nothing published on a failure)
        for (int32_t c = 0; c < nc; ++c) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "chunk join", err)) {
                return false;
            }
        }
        s.last_join_s    = since(t0);
        const int32_t nh = ch.half[1] > 0 ? 2 : 1;
        if (!pp_->check_errors(err) || !t.prompt_commit(ch.rows, nh, err) || !pp_->publish(pc, err)) {
            return false;
        }
        ++s.rounds;
        const int64_t pos = ch.start + ch.rows;
        if (t.slot().n != pos || (t.draft_on() && t.draft_n() != pos)) {
            return fail(err,
                        "prefill: chunk at " + std::to_string(ch.start) + " did not end at " + std::to_string(pos));
        }
        if (checkpoint_due(pos, to)) {
            const int32_t slot = book_.slot_for(pos, t.slot_index());
            if (!save(t, slot, err)) {
                book_.drop(slot);
                return false;
            }
            book_.record(slot, pos, token_key(tok_.data(), pos), t.slot_index());
            cs.checkpoint = slot;
            ++s.checkpoints;
        }
        cs.compute_s = since(tc);
        if (!next_up) {
            const auto tw = clk::now();
            join();
            if (!gather_ok_ || !upload(1 - sl, plan[k + 1].rows, err)) {
                return gather_ok_ ? false : fail(err, gather_err_);
            }
            cs.stall_s = since(tw);
        }
        s.chunks.push_back(cs);
        last = pc;
        if (progress && k + 1 < plan.size() && !progress(from, pos, to)) {
            // stopped (BG1): the chunk's publication and checkpoint finish before the caller forgets the slot's state
            for (int32_t c = 0; c < 2; ++c) {
                if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                    !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "stop", err)) {
                    return false;
                }
            }
            s.stopped = pos;
            return fail(err, "prefill: stopped at " + std::to_string(pos));
        }
    }
    // the handoff (FC-PF2): the last row's head (its logits retained), every row processed, the draft at N; TTTS1: a
    // front stage's ledger only (the head is the back's)
    if (m_->win.head ? !pp_->head_input(last, err) || !t.prompt_head(err) : !t.front_prompt_handoff(err)) {
        return false;
    }
    const ledger & e = t.slot();
    if (e.n != to || e.u != 0 || !e.retained || e.o != 0 || (t.draft_on() && t.draft_n() != to)) {
        return fail(err, "prefill: the handoff state is not N = L, U = 0, retained, O = 0 and the draft at N");
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "checkpoints", err)) {
            return false;
        }
    }
    return true;
}

bool prefiller::chunk_window(target & t, model::prompt_path & pp, model::prompt_chunk & c, std::string & err) {
    const model::window & w = m_->win;
    if (w.lo == 0 && w.head) {
        // the whole model (or split-test's one-box window): one chunk as before
        return w.whole() ? pp.chunk(c, err) :
                           pp.chunk_begin(c, err) && pp.chunk_layers(c, w.lo, w.hi + 1, err) && pp.chunk_end(c, err);
    }
    if (stage_ == nullptr) {
        return fail(err, "prefill: a split stage without its crossing (set_stage)");
    }
    if (!pp.chunk_begin(c, err) || (w.lo > 0 && !stage_->in(t, pp, c, err)) ||
        !pp.chunk_layers(c, w.lo, w.hi + 1, err)) {
        return false;
    }
    return w.head ? pp.chunk_end(c, err) : stage_->out(t, pp, c, err);
}

// ---- MU1: the multi-slot engine's resumable reads ----

bool prefiller::init_lane(target & t, std::string & err) {
    const model::reserved_class * st = m_->state_class("other-reader staging (MU1)");
    if (st == nullptr || m_->state_class("other-reader arena (MU1)") == nullptr) {
        return fail(err, "prefill: the other-reader lane's classes are not reserved (MU1)");
    }
    for (int32_t c = 0; c < 2; ++c) {
        lane_dev_[c] = reinterpret_cast<float *>(st->part_ptr(0, c));
        if (lane_dev_[c] == nullptr || st->plan.parts[0].bytes[c] < int64_t(model::lane_rows) * E * 4) {
            return fail(err, "prefill: the other-reader lane's staging is not reserved");
        }
    }
    void * h = nullptr;
    if (!cuda_ok(cudaHostAlloc(&h, size_t(model::lane_rows) * E * 4, cudaHostAllocPortable), "lane staging", err)) {
        return false;
    }
    lane_host_ = static_cast<float *>(h);
    pp2_ = std::make_unique<model::prompt_path>();
    model::pass_card * pc[2] = {&t.card(0), &t.card(1)};
    model::draft_card * dc[2] = {t.draft_on() ? &t.draft(0) : nullptr, t.draft_on() ? &t.draft(1) : nullptr};
    return pp2_->init(*m_, pc, dc, err, "other-reader arena (MU1)", model::lane_rows, model::lane_half_rows);
}

int32_t prefiller::lane_rows(int32_t lane) const {
    return lane == 0 ? chunk_rows_ : lane == 1 && pp2_ != nullptr ? model::lane_rows : 0;
}

bool prefiller::resident(target & t) const {
    return m_->pool != nullptr && !m_->single && !m_->kv_host && t.card(0).c.ex_service == nullptr;
}

int32_t prefiller::staged(int64_t id, int64_t chunk) const {
    for (int32_t k = 0; k < 2; ++k) {
        if (dev_tag_job_[k] == id && dev_tag_chunk_[k] == chunk) {
            return k;
        }
    }
    return -1;
}

bool prefiller::pump(std::string & err) {
    if (!gather_.joinable() || !gathered_.load(std::memory_order_acquire)) {
        return true;
    }
    join();
    const int32_t sl = gather_slot_;
    const int64_t job = gather_tag_job_, chunk = gather_tag_chunk_;
    gather_tag_job_ = gather_tag_chunk_ = -1;
    if (job < 0 || !gather_ok_ || sl == live_stage_) {
        return true; // forgotten or failed (the chunk's start gathers again and reports a failure)
    }
    if (!upload(sl, gather_rows_, err)) {
        return false;
    }
    dev_tag_job_[sl] = job;
    dev_tag_chunk_[sl] = chunk;
    return true;
}

bool prefiller::prefetch(target & t, const pf_job * now, const pf_job * next, std::string & err) {
    if (!resident(t)) {
        return true;
    }
    want_job_ = next != nullptr ? next->id : 0; // job ids start at 1: 0 follows nobody
    if (!pump(err)) {
        return false;
    }
    if (next == nullptr || !next->live || next->short_read) {
        return true;
    }
    // the job's next chunk: after the one in progress; after the one about to start when it follows itself
    size_t k = next->group >= 0 ? next->chunk + 1 : next->chunk;
    if (next == now && next->group < 0 &&
        (staged(next->id, int64_t(k)) >= 0 || (gather_tag_job_ == next->id && gather_tag_chunk_ == int64_t(k)))) {
        ++k;
    }
    if (k >= next->plan.size() || next->plan[k].rows > chunk_rows_ || staged(next->id, int64_t(k)) >= 0 ||
        (gather_tag_job_ == next->id && gather_tag_chunk_ == int64_t(k)) || gather_.joinable()) {
        return true; // nothing to stage, staged, being gathered, or the staging thread busy (a later call)
    }
    // the slot: not the lane-0 chunk's in progress, not the one holding the current turn's chunk about to start; an
    // empty one first
    const int32_t keep =
        now != nullptr && now->live && now->group < 0 ? staged(now->id, int64_t(now->chunk)) : int32_t(-1);
    int32_t sl = -1;
    for (int32_t s = 0; s < 2; ++s) {
        if (s != live_stage_ && s != keep && (sl < 0 || (dev_tag_job_[sl] >= 0 && dev_tag_job_[s] < 0))) {
            sl = s;
        }
    }
    if (sl < 0) {
        return true;
    }
    for (int32_t c = 0; c < 2; ++c) {
        // the pinned slot's last upload completed (never a wait here: a later call)
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err)) {
            return false;
        }
        const cudaError_t q = cudaEventQuery(static_cast<cudaEvent_t>(uploaded_[sl][c]));
        if (q == cudaErrorNotReady) {
            return true;
        }
        if (!cuda_ok(q, "host slot", err)) {
            return false;
        }
    }
    // the device slot's old rows are no longer staged: its next upload replaces them
    dev_tag_job_[sl] = dev_tag_chunk_[sl] = -1;
    start_gather(t.ple(), sl, next->plan[k], &next->tok);
    gather_tag_job_ = next->id;
    gather_tag_chunk_ = int64_t(k);
    gather_slot_ = sl;
    gather_rows_ = next->plan[k].rows;
    return true;
}

void prefiller::forget_job(int64_t id) {
    if (gather_tag_job_ == id) {
        join(); // the staging thread reads the job's tokens
        gather_tag_job_ = gather_tag_chunk_ = -1;
    }
    for (int32_t k = 0; k < 2; ++k) {
        if (dev_tag_job_[k] == id) {
            dev_tag_job_[k] = dev_tag_chunk_[k] = -1;
        }
    }
}

bool prefiller::job_start(target & t, pf_job & j, int32_t layers, pf_mode mode, std::string & err) {
    const pf_chunk & ch = j.plan[j.chunk];
    const int32_t lane = busy_[0] < 0 && ch.rows <= lane_rows(0)                      ? 0
                         : busy_[1] < 0 && pp2_ != nullptr && ch.rows <= lane_rows(1) ? 1
                                                                                      : -1;
    if (lane < 0) {
        return fail(err, "prefill: no lane free for slot " + std::to_string(j.slot) + "'s chunk of " +
                             std::to_string(ch.rows) + " rows (lane 0: slot " + std::to_string(busy_[0]) +
                             ", lane 1: slot " + std::to_string(busy_[1]) + ")");
    }
    if (layers < 1 || layers > fn::n_layer) {
        return fail(err, "prefill: a layer group of " + std::to_string(layers) + " layers");
    }
    model::prompt_path & pp = lane == 0 ? *pp_ : *pp2_;
    model::pass_card * pc[2] = {&t.card(0), &t.card(1)};
    model::draft_card * dc[2] = {t.draft_on() ? &t.draft(0) : nullptr, t.draft_on() ? &t.draft(1) : nullptr};
    pp.bind(pc, dc);
    const ple_params & pl = t.ple();
    const float * e[2] = {nullptr, nullptr};
    // MXMU1: with the expert service each chunk takes and returns its own loan (job_hold, job_release); on one card the
    // lent region holds PF1's staging and PB1's arena, which the pool refills between chunks
    model::expert_service * const xs = t.card(0).c.ex_service;
    const bool lent = xs != nullptr && xs->prompt_lent();
    const bool res = resident(t);
    if (!m_->win.ple()) {
        // TTTS1: a back stage stages no PLE rows (the chunk's e is unread)
        e[0]    = dev_[0][0];
        e[1]    = dev_[0][1];
        j.stage = -1;
    } else if (lane == 0) {
        // the running gather (another job's next chunk, or this one's): finished before its slots are reused. FMU1
        // (resident TP): only this chunk's gather is waited for; another job's keeps running into its own slot (a
        // finished one is uploaded and tagged)
        if (!res || (gather_tag_job_ == j.id && gather_tag_chunk_ == int64_t(j.chunk))) {
            join();
        } else if (!pump(err)) {
            return false;
        }
        const bool joined_mine = gather_tag_job_ == j.id && gather_tag_chunk_ == int64_t(j.chunk) && gather_ok_;
        int32_t d = staged(j.id, int64_t(j.chunk));
        bool up = false;
        if (d < 0 && joined_mine) {
            d = gather_slot_; // its rows are in host_[d]
            up = true;
        } else if (d < 0) {
            // FMU1: not the running gather's slot, nor (when free) the one staged for the job to follow
            d = !res                                                   ? 0
                : gather_.joinable()                                   ? 1 - gather_slot_
                : dev_tag_job_[0] >= 0 && dev_tag_job_[0] == want_job_ ? 1
                                                                       : 0;
            for (int32_t c = 0; c < 2; ++c) {
                if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                    !cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(uploaded_[d][c])), "host slot", err)) {
                    return false;
                }
            }
            if (!gather_rows(pl, m_->ple_table, j.tok, ch.start, ch.rows, host_[d], err)) {
                return false;
            }
            up = true;
        }
        // the loan before the staging slot's upload: no write enters the lent region while it is not held
        if ((xs != nullptr && !job_hold(t, pp, ch, d, j, err)) || (up && !upload(d, ch.rows, err))) {
            return false;
        }
        dev_tag_job_[d] = j.id;
        dev_tag_chunk_[d] = int64_t(j.chunk);
        if (!res || joined_mine) {
            gather_tag_job_ = gather_tag_chunk_ = -1;
        }
        for (int32_t c = 0; c < 2; ++c) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                !cuda_ok(cudaStreamWaitEvent(static_cast<cudaStream_t>(t.stream(c)),
                                             static_cast<cudaEvent_t>(uploaded_[d][c]), 0),
                         "wait staging", err)) {
                return false;
            }
            e[c] = dev_[d][c];
        }
        j.stage = d;
    } else {
        // the lane's one staging slot: its last reader (the lane's previous chunk) completed at that chunk's end.
        // FMU1 (resident TP): rows P staged in a main-lane slot are copied on the cards (the slot's readers recorded)
        const int32_t d = res ? staged(j.id, int64_t(j.chunk)) : int32_t(-1);
        if (d < 0 && !gather_rows(pl, m_->ple_table, j.tok, ch.start, ch.rows, lane_host_, err)) {
            return false;
        }
        for (int32_t c = 0; c < 2; ++c) {
            cudaStream_t s = static_cast<cudaStream_t>(t.stream(c));
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                (d < 0 &&
                 !cuda_ok(cudaMemcpyAsync(lane_dev_[c], lane_host_, size_t(ch.rows) * E * 4, cudaMemcpyHostToDevice, s),
                          "lane staging upload", err)) ||
                (d >= 0 &&
                 (!cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(uploaded_[d][c]), 0), "wait staging", err) ||
                  !cuda_ok(
                      cudaMemcpyAsync(lane_dev_[c], dev_[d][c], size_t(ch.rows) * E * 4, cudaMemcpyDeviceToDevice, s),
                      "lane staging copy", err) ||
                  !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(read_[d][c]), s), "record readers", err)))) {
                return false;
            }
            e[c] = lane_dev_[c];
        }
        if (d >= 0) {
            dev_tag_job_[d] = dev_tag_chunk_[d] = -1;
        }
        j.stage = -1;
    }
    // the lane's arena holds this job's predecessor and PLE history only when its last chunk there was this job's
    // previous one; otherwise they are rebuilt from the slot's committed state (run_start, both cards drained). MXMU1:
    // a lent arena is the pool's between chunks, so every chunk rebuilds them
    if (lent || cont_job_[lane] != j.id || cont_pos_[lane] != ch.start) {
        for (int32_t c = 0; c < 2; ++c) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "run start", err)) {
                return false;
            }
        }
        if (!pp.run_start(t.slot().cur, err)) {
            return false;
        }
    }
    cont_job_[lane] = -1; // a chunk in progress: its arena is no continuation until the chunk's end
    if (!t.prompt_begin(err)) {
        return false;
    }
    model::prompt_chunk & c = j.pc;
    c = model::prompt_chunk();
    c.index = static_cast<int64_t>(j.chunk);
    c.seq = seq_++;
    c.start = ch.start;
    c.rows = ch.rows;
    c.half[0] = ch.half[0];
    c.half[1] = ch.half[1];
    c.tok = j.tok.data() + ch.start;
    c.e[0] = e[0];
    c.e[1] = e[1];
    c.draft = t.draft_on();
    // MXMU1: the loan ends with the chunk, so nothing is staged ahead for the next one
    c.more = xs == nullptr && j.chunk + 1 < j.plan.size();
    j.lane = lane;
    j.group = 0;
    // FMU1 (resident TP): the chunk's geometry by mode (pf_mode); else MU1's groups of `layers` layers. TTTS1: over the
    // window's layers
    const int32_t n_win     = m_->win.hi - m_->win.lo + 1;
    j.layers    = !res || mode == pf_mode::shared ? std::min(layers, n_win) : mode == pf_mode::lone ? n_win : 1;
    j.end_group = res && mode == pf_mode::writers;
    j.groups    = (n_win + j.layers - 1) / j.layers + (j.end_group ? 1 : 0);
    busy_[lane] = j.slot;
    if (lane == 0) {
        live_stage_ = j.stage;
    }
    // FMU1 (resident TP): only while this job is the one to follow its turn (prefetch), and not over a running gather
    // (it stages another job's chunk in the other slot)
    const bool ahead = !res || ((want_job_ < 0 || want_job_ == j.id) && !gather_.joinable());
    if (lane == 0 && m_->win.ple() && j.chunk + 1 < j.plan.size() && ahead &&
        !(gather_tag_job_ == j.id && gather_tag_chunk_ == int64_t(j.chunk + 1))) {
        // the job's next chunk's rows gathered on the staging thread while this chunk computes (run_bulk's pipeline)
        const int32_t sl = 1 - j.stage;
        for (int32_t k = 0; k < 2; ++k) {
            if (!cuda_ok(cudaSetDevice(m_->device[k]), "device", err) ||
                !cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(uploaded_[sl][k])), "host slot", err)) {
                return false;
            }
        }
        if (dev_tag_job_[sl] == j.id && dev_tag_chunk_[sl] == int64_t(j.chunk + 1)) {
            return true; // already staged
        }
        if (res) {
            dev_tag_job_[sl] = dev_tag_chunk_[sl] = -1; // its next upload replaces the old rows
        }
        start_gather(pl, sl, j.plan[j.chunk + 1], &j.tok);
        gather_tag_job_ = j.id;
        gather_tag_chunk_ = int64_t(j.chunk + 1);
        gather_slot_ = sl;
        gather_rows_ = j.plan[j.chunk + 1].rows;
    }
    return true;
}

bool prefiller::job_finish_chunk(target & t, pf_job & j, bool & done, std::string & err) {
    done = false;
    const pf_chunk & ch = j.plan[j.chunk];
    model::prompt_path & pp = j.lane == 0 ? *pp_ : *pp2_;
    const int32_t nh = ch.half[1] > 0 ? 2 : 1;
    if (!pp.check_errors(err) || !t.prompt_commit(ch.rows, nh, err) || !pp.publish(j.pc, err)) {
        return false;
    }
    ++j.stats.rounds;
    const int64_t pos = ch.start + ch.rows;
    if (t.slot().n != pos || (t.draft_on() && t.draft_n() != pos)) {
        return fail(err, "prefill: chunk at " + std::to_string(ch.start) + " did not end at " + std::to_string(pos));
    }
    pf_chunk_stats cs;
    cs.c = ch;
    if (checkpoint_due(pos, j.to)) {
        const int32_t slot = book_.slot_for(pos, j.slot);
        if (!save(t, slot, err)) {
            book_.drop(slot);
            return false;
        }
        book_.record(slot, pos, token_key(j.tok.data(), pos), j.slot);
        cs.checkpoint = slot;
        ++j.checkpoints;
        ++j.stats.checkpoints;
    }
    // the publication's and the checkpoint's copies complete before the lane or the entry is reused (and before the
    // step returns: a step's work is complete). FMU1 (resident TP): stream order gives both, as in run_bulk: every
    // later reader of the slot's state, the lane's arena and the checkpoint (the job's next chunk, any slot's round, a
    // restore, the other lane's chunk) runs on the same compute streams, and a reclaim or trim drains them first
    // (work/FMU1/notes/wait-account.md, W10)
    const bool res = resident(t);
    for (int32_t c = 0; c < 2 && !res; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "chunk end", err)) {
            return false;
        }
    }
    model::expert_service * const xs = t.card(0).c.ex_service;
    const bool lent = xs != nullptr && xs->prompt_lent();
    if (j.lane == 0) {
        live_stage_ = -1; // the chunk's staging slot: its last readers recorded at the chunk's end
    }
    if (res) {
        // FMU1: the next chunk's gather is not waited for here (its start, or a finished gather's upload, takes it)
        if (!pump(err)) {
            return false;
        }
    } else if (j.lane == 0 && gather_tag_job_ == j.id && gather_tag_chunk_ == int64_t(j.chunk + 1)) {
        const auto tw = clk::now();
        join();
        if (!gather_ok_) {
            return fail(err, gather_err_);
        }
        // MXMU1: a lent staging slot is uploaded by the next chunk's job_start, under its loan (the rows stay in the
        // pinned host slot, tagged as the gather left them)
        if (!lent) {
            if (!upload(gather_slot_, j.plan[j.chunk + 1].rows, err)) {
                return false;
            }
            dev_tag_job_[gather_slot_] = j.id;
            dev_tag_chunk_[gather_slot_] = int64_t(j.chunk + 1);
            gather_tag_job_ = gather_tag_chunk_ = -1;
        }
        cs.stall_s = since(tw);
    }
    j.stats.chunks.push_back(cs);
    cont_job_[j.lane] = lent ? -1 : j.id;
    cont_pos_[j.lane] = pos;
    busy_[j.lane] = -1;
    const model::prompt_chunk last = j.pc;
    j.lane = -1;
    j.group = -1;
    ++j.chunk;
    ++j.chunks_done;
    j.committed = pos;
    if (j.chunk < j.plan.size()) {
        return job_release(t, err); // MXMU1: the chunk's loan back before any other slot's work
    }
    // the handoff (FC-PF2): the last row's head (its logits retained), every row processed, the draft at N
    model::prompt_path & hp = &pp == pp_.get() ? *pp_ : *pp2_;
    if (m_->win.head ? !hp.head_input(last, err) || !t.prompt_head(err) : !t.front_prompt_handoff(err)) {
        return false;
    }
    const ledger & e = t.slot();
    if (e.n != j.to || e.u != 0 || !e.retained || e.o != 0 || (t.draft_on() && t.draft_n() != j.to)) {
        return fail(err, "prefill: the handoff state is not N = L, U = 0, retained, O = 0 and the draft at N");
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "handoff", err)) {
            return false;
        }
    }
    done = true;
    return job_release(t, err); // MXMU1: after the head input, the arena's last reader
}

void prefiller::job_plan(int64_t from, int64_t to, bool cached, std::vector<pf_chunk> & plan, bool & short_read) const {
    // as run(): MX3's one-half chunks with the expert service (not the base arm); ME3's short read
    const int32_t split = cached && !legacy_ ? pf_cached_split_rows : pf_split_rows;
    plan = plan_chunks(from, to, legacy_ ? pf_chunk_rows : chunk_rows_, split);
    short_read = pf_short_read(from, to, short_read_, cached, legacy_);
}

bool prefiller::job_hold(target & t, model::prompt_path & pp, const pf_chunk & ch, int32_t d, pf_job & j,
                         std::string & err) {
    model::expert_service * const xs = t.card(0).c.ex_service;
    if (held_job_ >= 0) {
        return fail(err, "prefill: job " + std::to_string(held_job_) + "'s chunk holds the prompt loan");
    }
    // as run(), for this chunk: the arena laid out for it (a prefix), the spans it writes (its staging slot's rows and
    // the layout) borrowed; the base arm the whole region at the full layout. MX3: a one-half chunk stages through the
    // ring, sized by the chunk
    std::vector<model::byte_span> spans;
    const bool lent = xs->prompt_lent();
    if (lent) {
        const int32_t rc = legacy_ ? pf_chunk_rows : ch.rows;
        const int32_t rh = legacy_ ? pf_half_rows : std::max(ch.half[0], ch.half[1]);
        if (!pp.layout(rc, rh, err)) {
            return false;
        }
        for (int32_t c = 0; c < (m_->single ? 1 : 2); ++c) {
            spans.push_back({dev_[d][c], int64_t(ch.rows) * E * 4});
        }
        pp.arena_spans(spans);
    }
    xs->set_prompt_ring(ch.half[1] == 0 && bulk_ && !legacy_ ? ch.rows : 0);
    if (!pp.hold(true, err, lent && legacy_ ? nullptr : &spans)) {
        xs->set_prompt_ring(0);
        return false;
    }
    held_job_ = j.id;
    j.stats.layout_rows = std::max(j.stats.layout_rows, pp.layout_rows());
    j.stats.layout_half = std::max(j.stats.layout_half, pp.layout_half());
    int64_t loan = lent && legacy_ ? -1 : 0; // the largest chunk's loan (run's: -1 for the base arm's whole region)
    for (const model::byte_span & x : spans) {
        loan += lent && !legacy_ ? x.bytes : 0;
    }
    j.stats.loan_span_bytes = loan < 0 ? loan : std::max(j.stats.loan_span_bytes, loan);
    return true;
}

bool prefiller::job_release(target & t, std::string & err) {
    if (held_job_ < 0) {
        return true;
    }
    model::expert_service * const xs = t.card(0).c.ex_service;
    held_job_ = -1;
    if (xs != nullptr && xs->prompt_lent()) {
        // the pool refills the lent staging slots: no device slot holds a chunk's rows after the return
        dev_tag_job_[0] = dev_tag_job_[1] = dev_tag_chunk_[0] = dev_tag_chunk_[1] = -1;
    }
    return pp_->hold(false, err);
}

pf_step prefiller::job_step(target & t, pf_job & j, int32_t n_groups, int32_t layers, std::string & err, pf_mode mode) {
    const auto t0 = clk::now();
    if (!j.live || j.slot != t.slot_index()) {
        err = "prefill: no live job on slot " + std::to_string(t.slot_index());
        return pf_step::error;
    }
    if (j.group < 0) {
        if (j.cancel) {
            return pf_step::cancelled; // BG1: no chunk started; the completed ones and their checkpoints stay
        }
        if (j.short_read && j.chunk == 0) {
            // MXMU1: ME3's known-token short read in one step (the windows, no staging, arena or loan), as run()
            if (held_job_ >= 0) {
                err = "prefill: a short read while job " + std::to_string(held_job_) + "'s chunk holds the prompt loan";
                return pf_step::error;
            }
            tok_ = j.tok;
            j.stats.short_read = true;
            const bool ok = run_short(t, j.stats, err);
            const double dt = since(t0);
            ++j.steps;
            j.service_s += dt;
            j.longest_step_s = std::max(j.longest_step_s, dt);
            j.stats.wall_s = j.service_s;
            if (!ok) {
                job_abandon(t, j);
                return pf_step::error;
            }
            j.chunk = j.plan.size();
            j.chunks_done = static_cast<int32_t>(j.plan.size());
            j.committed = j.to;
            j.checkpoints = j.stats.checkpoints;
            return pf_step::done;
        }
        if (j.chunk >= j.plan.size() || !job_start(t, j, layers, mode, err)) {
            if (err.empty()) {
                err = "prefill: the job has no chunk left";
            }
            job_abandon(t, j);
            return pf_step::error;
        }
    }
    model::prompt_path & pp = j.lane == 0 ? *pp_ : *pp2_;
    // FMU1 (resident TP): a lone step runs the chunk's remaining groups and its finish; a writers-mode chunk's end
    // group and finish are steps of their own (FMU2: a step of several 1-layer groups stops before the end group)
    const bool lone = mode == pf_mode::lone && resident(t);
    const int32_t n = lone ? j.groups : std::max(1, n_groups);
    int32_t ran = 0;
    while (ran < n && j.group < j.groups) {
        const bool end_only = j.end_group && j.group == j.groups - 1;
        if (end_only && ran > 0 && !lone) {
            break;
        }
        // TTTS1: the window's layers [L0, L1); a back stage takes its crossing after chunk_begin, a front stage sends
        // its crossing at the end in place of the target's end and the draft
        const int32_t L0 = m_->win.lo, L1 = m_->win.hi + 1;
        const int32_t lo    = end_only ? L1 : L0 + j.group * j.layers;
        const int32_t hi    = end_only ? L1 : std::min(L1, lo + j.layers);
        const bool    end   = end_only || (!j.end_group && hi == L1);
        const bool    split = !(L0 == 0 && m_->win.head);
        if (split && stage_ == nullptr) {
            err = "prefill: a split stage without its crossing (set_stage)";
            job_abandon(t, j);
            return pf_step::error;
        }
        if ((j.group == 0 && (!pp.chunk_begin(j.pc, err) || (L0 > 0 && !stage_->in(t, pp, j.pc, err)))) ||
            (lo < hi && !pp.chunk_layers(j.pc, lo, hi, err)) ||
            (end && (m_->win.head ? !pp.chunk_end(j.pc, err) : !stage_->out(t, pp, j.pc, err)))) {
            job_abandon(t, j);
            return pf_step::error;
        }
        if (end && j.lane == 0 && j.stage >= 0) {
            for (int32_t c = 0; c < 2; ++c) {
                // the staging slot's last readers: the device slot is overwritten only after them (upload)
                if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                    !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(read_[j.stage][c]),
                                             static_cast<cudaStream_t>(t.stream(c))),
                             "record readers", err)) {
                    job_abandon(t, j);
                    return pf_step::error;
                }
            }
        }
        ++j.group;
        ++ran;
    }
    // the step's work complete on both cards: the compute streams and the lane's copy streams (TU3's completed step).
    // FMU1: a finish-only step (writers mode) has no groups to join
    for (int32_t c = 0; c < 2 && ran > 0; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "group join", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(pp.copy_stream(c))), "group join", err)) {
            job_abandon(t, j);
            return pf_step::error;
        }
    }
    j.groups_done += ran;
    pf_step out = pf_step::yielded;
    // FMU1: a writers-mode chunk's finish is its own step (the end group's step returns first), unless lone now
    if (j.group >= j.groups && (!j.end_group || ran == 0 || lone)) {
        bool done = false;
        if (!job_finish_chunk(t, j, done, err)) {
            job_abandon(t, j);
            return pf_step::error;
        }
        out = done ? pf_step::done : j.cancel ? pf_step::cancelled : pf_step::chunk_done;
    }
    const double dt = since(t0);
    ++j.steps;
    j.service_s += dt;
    j.longest_step_s = std::max(j.longest_step_s, dt);
    j.last_group_s = out == pf_step::yielded && ran > 0 ? dt / ran : -1.0;
    j.stats.wall_s = j.service_s;
    return out;
}

void prefiller::job_abandon(target & t, pf_job & j) {
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m_->device[c]);
        cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c)));
        cudaStreamSynchronize(static_cast<cudaStream_t>(copy_[c]));
    }
    if (gather_tag_job_ == j.id) {
        join();
        gather_tag_job_ = gather_tag_chunk_ = -1;
    }
    if (held_job_ == j.id) {
        // MXMU1: the chunk's loan returned (both cards drained above; prompt_hold drains again), so no loan, ring or
        // lent hold outlives the failed or abandoned chunk
        std::string e;
        if (!job_release(t, e)) {
            std::fprintf(stderr, "prefill: job %lld: the prompt loan was not returned: %s\n", (long long)j.id,
                         e.c_str());
        }
    }
    if (j.lane == 0) {
        live_stage_ = -1;
    }
    for (int32_t k = 0; k < 2; ++k) {
        if (dev_tag_job_[k] == j.id) {
            dev_tag_job_[k] = dev_tag_chunk_[k] = -1;
        }
        if (busy_[k] == j.slot) {
            busy_[k] = -1;
        }
        if (cont_job_[k] == j.id) {
            cont_job_[k] = -1;
        }
    }
    j.lane = -1;
    j.group = -1;
}

} // namespace fn::runtime
