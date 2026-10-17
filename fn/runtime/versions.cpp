#include "fn/runtime/versions.h"

#include "fn/kernels/round/round.h"
#include "fn/kernels/round/version.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <string>

namespace fn::runtime {

namespace {

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

} // namespace

void slot_versions::bind(int32_t card, int32_t device, void * stream, uint32_t * cur_dev) {
    device_[card] = device;
    stream_[card] = stream;
    cur_dev_[card] = cur_dev;
}

bool slot_versions::write_cur(uint32_t cur, std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        cudaStream_t s = static_cast<cudaStream_t>(stream_[c]);
        const uint32_t v = cur;
        if (cur_dev_[c] == nullptr || !cuda_ok(cudaSetDevice(device_[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaMemcpyAsync(cur_dev_[c], &v, 4, cudaMemcpyHostToDevice, s), "cur", err) ||
            !cuda_ok(cudaStreamSynchronize(s), "cur", err)) {
            return err.empty() ? fail(err, "cur: card not bound") : false;
        }
    }
    return true;
}

bool slot_versions::read_cur(uint32_t out[2], std::string & err) const {
    for (int32_t c = 0; c < 2; ++c) {
        cudaStream_t s = static_cast<cudaStream_t>(stream_[c]);
        if (!cuda_ok(cudaSetDevice(device_[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaMemcpyAsync(&out[c], cur_dev_[c], 4, cudaMemcpyDeviceToHost, s), "cur", err) ||
            !cuda_ok(cudaStreamSynchronize(s), "cur", err)) {
            return false;
        }
    }
    return true;
}

bool slot_versions::reset(ledger & l, std::string & err) {
    l = ledger();
    last_ = step_record();
    return write_cur(0, err);
}

void slot_versions::begin_step(ledger & l, bool head) {
    l.a = l.o; // the previous step is acknowledged by this one's start (RS1 section 3)
    last_ = step_record();
    last_.cur_pre = l.cur;
    last_.n_before = l.n;
    last_.l_before = static_cast<int64_t>(l.tok.size());
    last_.o_before = l.o;
    last_.pending_before = l.pending;
    last_.retained_before = l.retained;
    last_.retained_row_before = l.retained_row;
    last_.head = head;
}

bool slot_versions::committed(ledger & l, int32_t k, const uint32_t reported[2], std::string & err) {
    const uint32_t want = static_cast<uint32_t>(fn::round::version_slot(l.cur, k));
    if (k < 0 || k > fn::round::max_rows || reported[0] != want || reported[1] != want) {
        return fail(err, "commit: the cards' committed slots " + std::to_string(reported[0]) + " and " +
                             std::to_string(reported[1]) + " are not slot " + std::to_string(want) + " (k " +
                             std::to_string(k) + " from slot " + std::to_string(l.cur) + ")");
    }
    l.cur = want;
    last_.k = k;
    last_.kept = k;
    last_.valid = true;
    return true;
}

bool slot_versions::committed_prompt(ledger & l, int32_t k, const uint32_t reported[2], std::string & err) {
    if (!committed(l, k, reported, err)) {
        return false;
    }
    last_.kept   = 0;
    last_.prompt = true;
    return true;
}

bool slot_versions::restore(ledger & l, std::string & err) {
    if (!write_cur(l.cur, err)) {
        return false;
    }
    last_.k = 0;
    last_.kept = 0;
    last_.valid = true;
    if (last_.head) {
        l.retained = false; // the failed round's head overwrote the logits the retained row pointed into
        l.retained_row = -1;
    }
    settle(l);
    return true;
}

bool slot_versions::cut(ledger & l, int32_t j, int64_t n_keep, std::string & err) {
    if (!last_.valid) {
        return fail(err, "cut: no step to cut back within (deeper: a checkpoint restore and re-prefill, PF1 and SV1)");
    }
    const int64_t n_after = last_.n_before + j;
    const int64_t len = static_cast<int64_t>(l.tok.size());
    const int64_t o_keep = std::max<int64_t>(0, n_keep - l.prompt_len);
    if (j < 0 || j > last_.kept) {
        return fail(err, "cut: j " + std::to_string(j) + " outside 0 .. " + std::to_string(last_.kept));
    }
    if (n_keep < n_after || n_keep > len) {
        return fail(err, "cut: n_keep " + std::to_string(n_keep) + " outside " + std::to_string(n_after) + " .. " +
                             std::to_string(len));
    }
    if (o_keep < l.a) {
        return fail(err, "cut: n_keep drops an acknowledged output");
    }
    if (o_keep > 0 && n_keep - n_after != 1) {
        return fail(err, "cut: with outputs kept, exactly the pending token stays unprocessed (FC-SLOT1)");
    }
    const uint32_t cur = static_cast<uint32_t>(fn::round::version_slot(last_.cur_pre, j));
    if (!write_cur(cur, err)) {
        return false;
    }
    l.cur = cur;
    l.n = n_after;
    l.tok.resize(static_cast<size_t>(n_keep));
    l.prompt_len = std::min(l.prompt_len, n_keep); // a cut inside the prompt shortens it
    l.u = n_keep - n_after;
    l.o = o_keep;
    l.pending = o_keep > 0 ? l.tok[static_cast<size_t>(n_keep - 1)] : -1;
    if (!last_.head) {
        // a bootstrap step (k = 0, so j = 0) only read the retained logits: they still hold, so the decision repeats
        l.retained = last_.retained_before;
        l.retained_row = last_.retained_row_before;
    } else if (j == 0 || j < last_.k) {
        // the step's head overwrote the logits: the pre-step retained row is gone, and the kept rows' are not
        // reselected (SR1's choice), so nothing is retained until the last kept token is processed again
        l.retained = false;
        l.retained_row = -1;
    }
    last_.kept = j;
    settle(l);
    return true;
}

void slot_versions::settle(ledger & l) {
    const int32_t blocks = fn::qsa::blocks_complete(static_cast<int32_t>(l.n));
    fn::qsa::invalidate_from(l.summaries, blocks);
    if (l.summaries.n_valid < blocks) {
        fn::qsa::mark_updated(l.summaries, l.summaries.n_valid, blocks - l.summaries.n_valid);
    }
}

bool slot_versions::check(ledger & l, std::string & err) {
    ++l.checks;
    if (static_cast<int64_t>(l.tok.size()) != l.n + l.u) {
        ++l.failures;
        return fail(err, "ledger: L != N + U");
    }
    if (l.summaries.n_valid != fn::qsa::blocks_complete(static_cast<int32_t>(l.n)) ||
        l.cur >= static_cast<uint32_t>(fn::round::n_version) || l.a > l.o) {
        ++l.failures;
        return fail(err, "ledger: summaries' valid blocks != N / 4, cur out of range, or A > O");
    }
    return true;
}

} // namespace fn::runtime
