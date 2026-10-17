#pragma once

// BG2: the splice of a slot's generated answers back into a chat prompt that re-renders them. Plain token vectors and
// a caller-supplied comparison, so the walk runs without a model (tools/fnfl-splice-test).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

// an answer the slot generated and still holds: the slot's tokens [start, end); the model ended it with eog, which the
// prompt that re-renders it carries right after it
struct server_answer {
    size_t  start = 0;
    size_t  end   = 0;
    int32_t eog   = -1;
};

// one answer the walk reached: where the prompt left the slot, and whether that answer was spliced
struct server_splice_step {
    size_t index   = 0;  // into the answers
    size_t lcp     = 0;  // the prompt's common prefix with the slot when the walk reached the answer
    size_t n_gen   = 0;  // the answer's generated tokens
    size_t n_ren   = 0;  // the prompt's re-rendered copy (0 when the prompt has no eog after the answer's start)
    bool   spliced = false;
};

// whether the prompt's copy of an answer (ren) is the generated one (gen) in substance
using server_answer_same = std::function<bool(const std::vector<int32_t> & gen, const std::vector<int32_t> & ren)>;

// the slot keeps its first n tokens: drop the answers it no longer holds whole
void server_answers_keep_first(std::vector<server_answer> & answers, size_t n);

// the length of the common prefix of a and b, given that their first `from` tokens are equal
size_t server_common_prefix(const std::vector<int32_t> & a, const std::vector<int32_t> & b, size_t from = 0);

// The prompt `next` with the slot's generated tokens in place of each recorded answer it carries unchanged in
// substance. `answers` are ascending and inside `held`. The walk goes forward while the prompt leaves the slot inside a
// recorded answer (or at its eog); it stops where the prompt extends the slot, differs outside the answers, or carries
// an answer that differs. `steps` gets one entry per answer reached.
std::vector<int32_t> server_splice_answers(const std::vector<int32_t> &       held,
                                           const std::vector<int32_t> &       next,
                                           const std::vector<server_answer> & answers,
                                           const server_answer_same &         same,
                                           std::vector<server_splice_step> &  steps);
