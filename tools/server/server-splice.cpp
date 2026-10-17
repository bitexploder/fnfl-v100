#include "server-splice.h"

#include <algorithm>

void server_answers_keep_first(std::vector<server_answer> & answers, size_t n) {
    while (!answers.empty() && answers.back().end > n) {
        answers.pop_back();
    }
}

size_t server_common_prefix(const std::vector<int32_t> & a, const std::vector<int32_t> & b, size_t from) {
    const size_t n = std::min(a.size(), b.size());
    size_t       i = std::min(from, n);
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}

std::vector<int32_t> server_splice_answers(const std::vector<int32_t> &       held,
                                           const std::vector<int32_t> &       next,
                                           const std::vector<server_answer> & answers,
                                           const server_answer_same &         same,
                                           std::vector<server_splice_step> &  steps) {
    std::vector<int32_t> out = next;
    size_t               lcp = server_common_prefix(held, out);
    for (size_t i = 0; i < answers.size(); ++i) {
        if (lcp >= held.size() || lcp >= out.size()) {
            break;  // an extension (or a prefix) of the slot
        }
        const server_answer & a = answers[i];
        if (a.start >= a.end || a.end > held.size()) {
            break;  // not a record of this slot's tokens
        }
        if (lcp > a.end) {
            continue;  // the prompt carries this answer and its eog as the slot holds them
        }
        if (lcp < a.start) {
            break;  // the prompt changed before this answer
        }
        server_splice_step st;
        st.index     = i;
        st.lcp       = lcp;
        st.n_gen     = a.end - a.start;
        const auto e = std::find(out.begin() + a.start, out.end(), a.eog);
        if (e == out.end()) {
            steps.push_back(st);
            break;
        }
        const std::vector<int32_t> gen(held.begin() + a.start, held.begin() + a.end);
        const std::vector<int32_t> ren(out.begin() + a.start, e);
        st.n_ren = ren.size();
        if (!same(gen, ren)) {
            steps.push_back(st);
            break;
        }
        std::vector<int32_t> spliced(out.begin(), out.begin() + a.start);
        spliced.insert(spliced.end(), gen.begin(), gen.end());
        spliced.insert(spliced.end(), e, out.end());
        out        = std::move(spliced);
        lcp        = server_common_prefix(held, out, a.start);
        st.spliced = true;
        steps.push_back(st);
    }
    return out;
}
