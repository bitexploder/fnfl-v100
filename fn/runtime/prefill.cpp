#include "fn/runtime/prefill.h"

#include "fn/kernels/ple/ple.h"
#include "fn/model/pass.h"
#include "fn/model/state.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

namespace fn::runtime {

namespace {

constexpr int32_t E = ple_input; // 2,560 PLE input values a row
constexpr int64_t gdn_bytes = int64_t(model::gdn_heads_card) * model::gdn_head_dim * model::gdn_head_dim * 4;
constexpr int64_t conv_bytes = int64_t(model::gdn_conv_channels_card) * model::gdn_conv_history * 4;
constexpr int64_t ple_hist_bytes = fn::ple::hist_version_floats * 4;
constexpr int64_t carry_bytes = int64_t(4) * 1280 * 4;
static_assert(pf_max_checkpoints == model::prefill_checkpoints, "the checkpoint class holds pf_max_checkpoints");
static_assert(pf_chunk_rows == model::prefill_stage_rows, "a staging slot holds one chunk");
static_assert(pf_chunk_rows == 2 * pf_half_rows && pf_chunk_rows <= pf_span, "FC-PF3's chunk and halves");
static_assert(pf_split_rows <= pf_half_rows && pf_half_rows % 64 == 0, "a split chunk's halves");
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

std::vector<pf_chunk> plan_chunks(int64_t from, int64_t to) {
    std::vector<pf_chunk> out;
    int64_t pos = std::max<int64_t>(from, 0);
    while (pos < to) {
        // the tiling restarts at every multiple of 8,192 and at from
        const int64_t span_end = (pos / pf_span + 1) * pf_span;
        const int64_t end = std::min({pos + pf_chunk_rows, span_end, to});
        pf_chunk c;
        c.start = pos;
        c.rows = static_cast<int32_t>(end - pos);
        // up to pf_split_rows one half; above, two near-equal halves, the first a multiple of 64 rows (a GDN block)
        c.half[0] = c.rows <= pf_split_rows ? c.rows : std::min(pf_half_rows, ((c.rows + 1) / 2 + 63) / 64 * 64);
        c.half[1] = c.rows - c.half[0];
        out.push_back(c);
        pos = end;
    }
    return out;
}

bool checkpoint_due(int64_t pos, int64_t prompt_end) {
    return pos > 0 && (pos % pf_span == 0 || pos == prompt_end);
}

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

int32_t checkpoint_book::slot_for(int64_t pos) const {
    int32_t empty = -1, oldest = 0;
    for (int32_t i = 0; i < pf_max_checkpoints; ++i) {
        if (e_[i].pos == pos) {
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

void checkpoint_book::record(int32_t slot, int64_t pos, uint64_t key) {
    e_[slot].pos = pos;
    e_[slot].key = key;
    e_[slot].age = next_age_++;
}

int32_t checkpoint_book::latest(int64_t n) const {
    int32_t best = -1;
    for (int32_t i = 0; i < pf_max_checkpoints; ++i) {
        if (e_[i].pos >= 0 && e_[i].pos <= n && (best < 0 || e_[i].pos > e_[best].pos)) {
            best = i;
        }
    }
    return best;
}

void checkpoint_book::drop(int32_t slot) {
    e_[slot] = pf_checkpoint();
}

void checkpoint_book::drop_after(int64_t n) {
    for (pf_checkpoint & c : e_) {
        if (c.pos > n) {
            c = pf_checkpoint();
        }
    }
}

void checkpoint_book::clear() {
    for (pf_checkpoint & c : e_) {
        c = pf_checkpoint();
    }
}

int32_t checkpoint_book::count() const {
    int32_t n = 0;
    for (const pf_checkpoint & c : e_) {
        n += c.pos >= 0 ? 1 : 0;
    }
    return n;
}

int32_t choose_checkpoint(checkpoint_book & book, const std::vector<int32_t> & all, int64_t n_reuse) {
    // at least the last token is processed, so its logits exist (FC-PF1's rule for nothing new)
    const int64_t keep = std::min<int64_t>(n_reuse, static_cast<int64_t>(all.size()) - 1);
    int32_t slot = -1;
    while (keep > 0 && (slot = book.latest(keep)) >= 0) {
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
    ck_bytes_ = model::checkpoint_bytes(sp);
    if (ck_bytes_ != model::n_gdn_layers * (gdn_bytes + conv_bytes) + ple_hist_bytes + carry_bytes) {
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
    return pp_->init(m, pc, dc, err);
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

void prefiller::start_gather(const ple_params & p, int32_t s, const pf_chunk & c) {
    gathered_.store(false, std::memory_order_relaxed);
    gather_ok_ = false;
    gather_err_.clear();
    gather_ = std::thread([this, &p, s, c] {
        const auto t0 = clk::now();
        gather_ok_ = gather_rows(p, m_->ple_table, tok_, c.start, c.rows, host_[s], gather_err_);
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
    for (int32_t c = 0; c < 2; ++c) {
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
            if (!cuda_ok(cudaMemcpyAsync(q + off, reinterpret_cast<const uint8_t *>(st.gdn) + cur * gdn_bytes,
                                         size_t(gdn_bytes), cudaMemcpyDeviceToDevice, s),
                         "checkpoint GDN state", err) ||
                !cuda_ok(cudaMemcpyAsync(q + off + gdn_bytes,
                                         reinterpret_cast<const uint8_t *>(st.conv) + cur * conv_bytes,
                                         size_t(conv_bytes), cudaMemcpyDeviceToDevice, s),
                         "checkpoint GDN history", err)) {
                return false;
            }
            off += gdn_bytes + conv_bytes;
        }
        if (gdn != model::n_gdn_layers ||
            !cuda_ok(cudaMemcpyAsync(q + off, reinterpret_cast<const uint8_t *>(p.ple_hist) + cur * ple_hist_bytes,
                                     size_t(ple_hist_bytes), cudaMemcpyDeviceToDevice, s),
                     "checkpoint PLE history", err)) {
            return err.empty() ? fail(err, "prefill: not 36 GDN layers") : false;
        }
        off += ple_hist_bytes;
        if (t.draft_on() &&
            !cuda_ok(cudaMemcpyAsync(q + off, t.draft(c).carry_prev, size_t(carry_bytes), cudaMemcpyDeviceToDevice, s),
                     "checkpoint retained carry", err)) {
            return false;
        }
    }
    return true;
}

bool prefiller::load(target & t, int32_t slot, std::string & err) {
    const uint32_t cur = t.slot().cur; // 0 after rewind
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
            if (!cuda_ok(cudaMemcpyAsync(reinterpret_cast<uint8_t *>(st.gdn) + cur * gdn_bytes, q + off,
                                         size_t(gdn_bytes), cudaMemcpyDeviceToDevice, s),
                         "restore GDN state", err) ||
                !cuda_ok(cudaMemcpyAsync(reinterpret_cast<uint8_t *>(st.conv) + cur * conv_bytes, q + off + gdn_bytes,
                                         size_t(conv_bytes), cudaMemcpyDeviceToDevice, s),
                         "restore GDN history", err)) {
                return false;
            }
            off += gdn_bytes + conv_bytes;
        }
        if (!cuda_ok(cudaMemcpyAsync(reinterpret_cast<uint8_t *>(p.ple_hist) + cur * ple_hist_bytes, q + off,
                                     size_t(ple_hist_bytes), cudaMemcpyDeviceToDevice, s),
                     "restore PLE history", err)) {
            return false;
        }
        off += ple_hist_bytes;
        if ((t.draft_on() &&
             !cuda_ok(cudaMemcpyAsync(t.draft(c).carry_prev, q + off, size_t(carry_bytes), cudaMemcpyDeviceToDevice, s),
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
    const int32_t slot = book_.slot_for(l.n);
    if (!save(t, slot, err)) {
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "checkpoint", err)) {
            return false;
        }
    }
    book_.record(slot, l.n, token_key(l.tok.data(), l.n));
    return true;
}

bool prefiller::begin(target & t, const std::vector<int32_t> & all, int64_t n_reuse, int64_t & from,
                      std::string & err) {
    from = 0;
    if (all.empty() || n_reuse < 0) {
        return fail(err, "prefill: no tokens");
    }
    const int32_t slot = choose_checkpoint(book_, all, n_reuse);
    if (slot < 0) {
        book_.clear();
        if (!t.reset(err)) {
            return false;
        }
        t.set_prompt(all);
        return true;
    }
    const int64_t c = book_.at(slot).pos;
    if (!t.rewind(err) || !load(t, slot, err) || !t.restored(c, all, err)) {
        book_.clear(); // the slot's state is unknown
        return false;
    }
    book_.drop_after(c); // the re-prefill rewrites the state past c
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
    const std::vector<pf_chunk> plan = plan_chunks(from, to);
    tok_ = l.tok;
    pf_run_stats local;
    pf_run_stats & s = st != nullptr ? *st : local;
    s = pf_run_stats();
    s.from = from;
    s.to = to;
    s.t0             = t0;
    // the staging thread is joined on every exit
    struct joiner {
        prefiller * p;
        ~joiner() { p->join(); }
    } jn{this};
    const ple_params & pp = t.ple();
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t k = 0; k < 2; ++k) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                !cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(uploaded_[k][c])), "host slot", err)) {
                return false;
            }
        }
    }
    start_gather(pp, 0, plan[0]);
    join();
    if (!gather_ok_) {
        return fail(err, gather_err_);
    }
    if (!upload(0, plan[0].rows, err)) {
        return false;
    }
    s.bulk        = bulk_;
    const bool ok = bulk_ ? run_bulk(t, s, plan, progress, t0, err) : run_forced(t, s, plan, progress, err);
    s.wall_s      = since(t0);
    if (ok) {
        s.ready_s = s.wall_s;
        return true;
    }
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
            const int32_t slot = book_.slot_for(pos);
            if (!save(t, slot, err)) {
                book_.drop(slot);
                return false;
            }
            book_.record(slot, pos, token_key(tok_.data(), pos));
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

bool prefiller::run_bulk(target &                      t,
                         pf_run_stats &                s,
                         const std::vector<pf_chunk> & plan,
                         const prefill_progress &      progress,
                         clk::time_point               t0,
                         std::string &                 err) {
    const int64_t      from = s.from, to = s.to;
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
        cs.gather_s  = gather_s_;
        bool next_up = k + 1 >= plan.size();
        if (!next_up) {
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
        if (!t.prompt_begin(err) || !pp_->chunk(pc, err)) {
            return false;
        }
        for (int32_t c = 0; c < 2; ++c) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
                !cuda_ok(
                    cudaEventRecord(static_cast<cudaEvent_t>(read_[sl][c]), static_cast<cudaStream_t>(t.stream(c))),
                    "record readers", err)) {
                return false;
            }
        }
        // the chunk's end: both cards joined, then its errors, then its publication (nothing published on a failure)
        for (int32_t c = 0; c < 2; ++c) {
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
            const int32_t slot = book_.slot_for(pos);
            if (!save(t, slot, err)) {
                book_.drop(slot);
                return false;
            }
            book_.record(slot, pos, token_key(tok_.data(), pos));
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
    // the handoff (FC-PF2): the last row's head (its logits retained), every row processed, the draft at N
    if (!pp_->head_input(last, err) || !t.prompt_head(err)) {
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

} // namespace fn::runtime
