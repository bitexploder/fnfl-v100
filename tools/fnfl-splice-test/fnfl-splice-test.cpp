// fnfl-splice-test: the walk that splices a slot's generated answers back into a chat prompt (BG2), without a model.
//
// A toy vocabulary stands in for the tokenizer: the comparison is the answers' text, so a token split ("HA" + "IK"
// against "H" + "AI" + "K") is the same answer and a changed word is not. The conversation is three turns: two recorded
// answers in the slot, and a third prompt that re-renders both.
// usage: fnfl-splice-test. Exit status: 0 when every case passes.

#include "server-splice.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {

using tokens = std::vector<int32_t>;

enum : int32_t { U = 0, A = 1, EOG = 2, Q1 = 3, Q2 = 4, Q3 = 5, Q9 = 6 };

const std::map<int32_t, std::string> vocab = {
    { U,   "<u>" },
    { A,   "<a>" },
    { EOG, "<e>" },
    { Q1,  "q1"  },
    { Q2,  "q2"  },
    { Q3,  "q3"  },
    { Q9,  "q9"  },
    { 10,  "HA"  },
    { 11,  "IK"  },
    { 12,  "H"   },
    { 13,  "AI"  },
    { 14,  "K"   },
    { 20,  "foo" },
    { 21,  "bar" },
    { 22,  "baz" },
    { 25,  "BAR" },
    { 26,  "ba"  },
    { 27,  "z"   },
    { 28,  "qux" },
};

std::string text(const tokens & t) {
    std::string s;
    for (const int32_t id : t) {
        s += vocab.at(id);
    }
    return s;
}

tokens cat(std::initializer_list<tokens> parts) {
    tokens out;
    for (const tokens & p : parts) {
        out.insert(out.end(), p.begin(), p.end());
    }
    return out;
}

std::string show(const tokens & t) {
    std::string s;
    for (const int32_t id : t) {
        s += (s.empty() ? "" : " ") + vocab.at(id);
    }
    return s;
}

// the conversation: turn 1's prompt, its answer as the model split it and as the tokenizer splits it; turn 2's tail
const tokens p1     = { U, Q1, A };
const tokens a1_gen = { 10, 11, 20 };      // "HAIKfoo", the model's split
const tokens a1_ren = { 12, 13, 14, 20 };  // "HAIKfoo", the tokenizer's
const tokens u2     = { EOG, U, Q2, A };
const tokens a2_gen = { 21, 22 };          // "barbaz"
const tokens u3     = { EOG, U, Q3, A };

// the slot after turn 2 (turn 2's prompt was spliced, so it holds a1_gen), and its two answer records
const tokens                     held    = cat({ p1, a1_gen, u2, a2_gen });
const std::vector<server_answer> answers = {
    { p1.size(),                             p1.size() + a1_gen.size(), EOG },
    { p1.size() + a1_gen.size() + u2.size(), held.size(),               EOG },
};

int n_fail = 0;

struct expect_step {
    size_t index;
    bool   spliced;
};

void check(const char *                       name,
           const tokens &                     held_c,
           const std::vector<server_answer> & answers_c,
           const tokens &                     next,
           const tokens &                     want,
           const std::vector<expect_step> &   want_steps) {
    std::vector<server_splice_step> steps;
    const server_answer_same        same = [](const tokens & gen, const tokens & ren) {
        return text(gen) == text(ren);
    };
    const tokens out = server_splice_answers(held_c, next, answers_c, same, steps);
    bool         ok  = out == want && steps.size() == want_steps.size();
    for (size_t i = 0; ok && i < steps.size(); ++i) {
        ok = steps[i].index == want_steps[i].index && steps[i].spliced == want_steps[i].spliced;
    }
    std::printf("%s %s: %zu steps, prompt %zu -> %zu, common prefix with the slot %zu of %zu\n", ok ? "PASS" : "FAIL",
                name, steps.size(), next.size(), out.size(), server_common_prefix(held_c, out), held_c.size());
    if (!ok) {
        std::printf("  got  %s\n  want %s\n", show(out).c_str(), show(want).c_str());
        for (const server_splice_step & s : steps) {
            std::printf("  step: answer %zu at %zu, %zu generated, %zu re-rendered, spliced %d\n", s.index, s.lcp,
                        s.n_gen, s.n_ren, (int) s.spliced);
        }
        ++n_fail;
    }
}

}  // namespace

int main() {
    // BG2's case: a split in turn 1's answer, turn 2's answer re-rendered as generated. The walk splices answer 0 and
    // turn 3 extends the slot.
    check("split in the first answer", held, answers,
          cat({
              p1, a1_ren, u2, a2_gen, u3
    }),
          cat({ held, u3 }), { { 0, true } });

    // a split in both answers: both spliced
    check("split in both answers", held, answers,
          cat({
              p1, a1_ren, u2, { 21, 26, 27 },
                 u3
    }),
          cat({ held, u3 }), { { 0, true }, { 1, true } });

    // the client edited the second answer: the first is spliced, the second keeps its re-render (a re-prefill from it)
    check("edit in the second answer", held, answers,
          cat({
              p1, a1_ren, u2, { 25, 22 },
                 u3
    }),
          cat({ p1, a1_gen, u2, { 25, 22 }, u3 }), { { 0, true }, { 1, false } });

    // the client edited the first answer: not spliced, and the walk stops there
    check("edit in the first answer", held, answers,
          cat({
              p1, { 12, 13, 14, 28 },
               u2, a2_gen, u3
    }),
          cat({ p1, { 12, 13, 14, 28 }, u2, a2_gen, u3 }), { { 0, false } });

    // the client changed turn 1's question, before both answers: nothing to splice
    check("edit before both answers", held, answers,
          cat({
              { U, Q9, A },
              a1_ren, u2, a2_gen, u3
    }),
          cat({ { U, Q9, A }, a1_ren, u2, a2_gen, u3 }), {});

    // the client changed turn 2's question, between the answers: the first is spliced, the walk stops at the change
    check("edit between the answers", held, answers,
          cat({
              p1, a1_ren, { EOG, U, Q9, A },
                a2_gen, u3
    }),
          cat({ p1, a1_gen, { EOG, U, Q9, A }, a2_gen, u3 }), { { 0, true } });

    // BG1's case: only the last answer split; the first matches token for token and is passed over
    check("split in the last answer only", held, answers,
          cat({
              p1, a1_gen, u2, { 21, 26, 27 },
                 u3
    }),
          cat({ held, u3 }), { { 1, true } });

    // already an extension: nothing to do
    check("extension", held, answers, cat({ held, u3 }), cat({ held, u3 }), {});

    // the prompt stops inside the first answer, with no end-of-generation after it: not spliced
    check("no end after the answer", held, answers,
          cat({
              p1, a1_ren
    }),
          cat({ p1, a1_ren }), { { 0, false } });

    // no records (a new slot, or answers that did not end at end-of-generation): today's behaviour
    check("no records", held, {}, cat({ p1, a1_ren, u2, a2_gen, u3 }), cat({ p1, a1_ren, u2, a2_gen, u3 }), {});

    // the slot's tokens truncated inside the second answer: its record goes, the first stays
    {
        std::vector<server_answer> kept = answers;
        server_answers_keep_first(kept, answers[1].end - 1);
        const bool ok = kept.size() == 1 && kept[0].end == answers[0].end;
        server_answers_keep_first(kept, answers[0].start);
        const bool ok0 = kept.empty();
        std::printf("%s keep_first drops the answers the slot no longer holds whole\n", ok && ok0 ? "PASS" : "FAIL");
        n_fail += ok && ok0 ? 0 : 1;
    }

    std::printf("fnfl-splice-test: %s\n", n_fail == 0 ? "all PASS" : "FAIL");
    return n_fail == 0 ? 0 : 1;
}
