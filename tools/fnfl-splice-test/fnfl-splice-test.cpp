// fnfl-splice-test: the walk that splices a slot's generated answers back into a chat prompt (BG2), without a model.
//
// A toy vocabulary stands in for the tokenizer: the comparison is the answers' text, so a token split ("HA" + "IK"
// against "H" + "AI" + "K") is the same answer and a changed word is not. The conversation is three turns: two recorded
// answers in the slot, and a third prompt that re-renders both.
// MU1: also the two-user scheduling policy's pure parts (server-tu.h, Superfast TU1..TU6's rules): the readers' order,
// the remaining and uncached counts, the one-sync slice's group prediction and the idle LRU's tie.
// usage: fnfl-splice-test. Exit status: 0 when every case passes.

#include "server-splice.h"
#include "server-tu.h"

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

    // ---- MU1: two-user scheduling (server-tu.h) ----
    const auto order = [](std::vector<tu_reader> r, int32_t fast) {
        tu_sort(r, fast);
        std::vector<int> out;
        for (const tu_reader & x : r) {
            out.push_back(x.slot);
        }
        return out;
    };
    const auto put = [&](const char * name, bool ok, const std::string & got) {
        std::printf("%s TU %s%s%s\n", ok ? "PASS" : "FAIL", name, got.empty() ? "" : ": ", got.c_str());
        n_fail += ok ? 0 : 1;
    };
    const auto list = [](const std::vector<int> & v) {
        std::string s;
        for (const int x : v) {
            s += (s.empty() ? "" : ",") + std::to_string(x);
        }
        return s;
    };
    const int32_t fast = 4096; // the reviewed TU6 configuration's n_batch
    {
        // unequal priority: the higher first even when it is long and the other short
        std::vector<tu_reader> r(2);
        r[0] = { 0, 0, 100, 100, 0, 10, false };
        r[1] = { 1, 5, 60000, 60000, 0, 11, false };
        const std::vector<int> o = order(r, fast);
        put("unequal priority", o == std::vector<int>{ 1, 0 }, list(o));
    }
    {
        // short readers first; among them fewer remaining, then the task ID
        std::vector<tu_reader> r(4);
        r[0] = { 0, 0, 60000, 60000, 1, 10, true };
        r[1] = { 1, 0, 3000, 3000, 0, 12, false };
        r[2] = { 2, 0, 2000, 2000, 0, 13, false };
        r[3] = { 3, 0, 2000, 2000, 0, 11, false };
        const std::vector<int> o = order(r, fast);
        put("short first, fewer remaining, then task ID", o == std::vector<int>{ 3, 2, 1, 0 }, list(o));
    }
    {
        // the threshold's edges: uncached = fast is short, fast + 1 long
        std::vector<tu_reader> r(2);
        r[0] = { 0, 0, fast + 1, fast + 1, 0, 1, false };
        r[1] = { 1, 0, fast, fast, 0, 2, false };
        const std::vector<int> o = order(r, fast);
        put("short threshold edge (uncached <= n_batch)", o == std::vector<int>{ 1, 0 }, list(o));
    }
    {
        // a warm read: few remaining tokens but a long uncached read is still long (TU6 classifies by uncached)
        const int32_t n_tokens = 70000, prefix = 0;
        const int32_t rem_cold = tu_remaining_tokens(false, n_tokens, prefix, 0);
        const int32_t rem_begun = tu_remaining_tokens(true, n_tokens, prefix, 68000);
        const int32_t unc = tu_uncached(rem_begun, n_tokens, prefix);
        const int32_t rem_warm = tu_remaining_tokens(false, n_tokens, 67000, 0);
        const int32_t unc_warm = tu_uncached(rem_warm, n_tokens, 67000);
        const bool ok = rem_cold == 70000 && rem_begun == 2000 && unc == 70000 && rem_warm == 3000 && unc_warm == 3000;
        put("remaining from the committed tokens, uncached from the original read", ok,
            std::to_string(rem_cold) + " " + std::to_string(rem_begun) + " " + std::to_string(unc) + " " +
                std::to_string(rem_warm) + " " + std::to_string(unc_warm));
    }
    {
        // active-reader ownership: among equal-priority long readers the live job's reader keeps its turn, even with
        // a later last turn and more remaining
        std::vector<tu_reader> r(3);
        r[0] = { 0, 0, 20000, 20000, 1, 10, false };
        r[1] = { 1, 0, 60000, 60000, 7, 11, true };
        r[2] = { 2, 0, 30000, 30000, 3, 12, false };
        const std::vector<int> o = order(r, fast);
        put("the live reader keeps its turn", o == std::vector<int>{ 1, 0, 2 }, list(o));
    }
    {
        // equal-priority rotation: with no live job the earliest last turn goes first, a tie by task ID; after its turn
        // (its seq advanced) the next one leads
        std::vector<tu_reader> r(4);
        r[0] = { 0, 0, 30000, 32768, 5, 20, false };
        r[1] = { 1, 0, 30000, 32768, 6, 21, false };
        r[2] = { 2, 0, 30000, 32768, 0, 23, false };
        r[3] = { 3, 0, 30000, 32768, 0, 22, false };
        std::vector<int> o = order(r, fast);
        bool ok = o == std::vector<int>{ 3, 2, 0, 1 };
        r[3].last_seq = 7;
        o = order(r, fast);
        ok = ok && o == std::vector<int>{ 2, 0, 1, 3 };
        put("equal-priority chunk rotation and task-ID tie", ok, list(o));
    }
    {
        // the one-sync slice: no estimate runs one group and ends; with an estimate floor((budget - t) / g), at least
        // one, ending once a step ran and none fits; the estimate counts the step's completed groups only
        const bool a = tu_next_groups(100.0, 0.0, -1.0, 0) == 1 && tu_next_groups(100.0, 40.0, -1.0, 1) == 0;
        const bool b = tu_next_groups(100.0, 0.0, 30.0, 0) == 3 && tu_next_groups(100.0, 90.0, 30.0, 1) == 0;
        const bool c = tu_next_groups(100.0, 0.0, 190.0, 0) == 1 && tu_next_groups(100.0, 190.0, 190.0, 1) == 0;
        const bool d = tu_next_groups(100.0, 0.0, 1e-6, 0) == 1024;
        // a step that queued 4 groups but completed 2 in 60 ms: the next estimate is 30 ms a group (completed work)
        const double est = 60.0 / 2;
        const bool e = tu_next_groups(100.0, 60.0, est, 1) == 1;
        put("slice groups from completed work", a && b && c && d && e,
            std::to_string(a) + std::to_string(b) + std::to_string(c) + std::to_string(d) + std::to_string(e));
    }
    {
        // TU6's idle LRU: ties go to the lowest slot; an erased slot (t_last_used -1) first; busy slots skipped
        std::vector<tu_slot_view> v(4);
        v[0] = { true, 50 };
        v[1] = { true, 50 };
        v[2] = { false, 10 };
        v[3] = { true, 70 };
        const int a = tu_lru(v);
        v[3].t_last_used = -1;
        const int b = tu_lru(v);
        v[0].available = v[1].available = v[3].available = false;
        const int c = tu_lru(v);
        put("idle LRU tie, erase and busy slots", a == 0 && b == 3 && c == -1,
            std::to_string(a) + " " + std::to_string(b) + " " + std::to_string(c));
    }

    // ---- FMU1: the presence rule and the reader to follow (server-tu.h) ----
    {
        // the path from what is active: nobody else lone, any writer writers, only other readers shared
        const bool a = tu_presence(0, 0) == tu_mode::lone && tu_presence(1, 0) == tu_mode::writers &&
                       tu_presence(3, 2) == tu_mode::writers && tu_presence(0, 1) == tu_mode::shared;
        // FMU2's reader rule: every mode has the whole slice; beside writers one step of main's group in 1-layer
        // groups (the policy's 2 layers at a chunk's start or when a group overfills the slice, else as many as fill
        // it, at most 1,024)
        const bool b = tu_mode_budget(tu_mode::writers, 100.0) == 100.0 &&
                       tu_mode_budget(tu_mode::shared, 100.0) == 100.0 && tu_writers_groups(100.0, -1.0, 2) == 2 &&
                       tu_writers_groups(100.0, 42.0, 2) == 2 && tu_writers_groups(100.0, 20.0, 2) == 5 &&
                       tu_writers_groups(100.0, 120.0, 2) == 2 && tu_writers_groups(100.0, 0.0, 2) == 1024 &&
                       tu_writers_groups(100.0, -1.0, 0) == 1;
        put("presence rule: lone, writers, shared; the reader rule's whole slice and main's group", a && b,
            std::to_string(a) + std::to_string(b));
    }
    {
        // a lone writer: one generating slot and no reader, admission or job; idle slots holding prefixes cost nothing
        const bool ok = tu_lone_writer(1, 0, false) && !tu_lone_writer(2, 0, false) && !tu_lone_writer(1, 1, false) &&
                        !tu_lone_writer(1, 0, true) && !tu_lone_writer(0, 0, false);
        // admissions: all when nothing else has model work, one an iteration otherwise
        const bool adm = tu_admit_more(false, 3) && tu_admit_more(true, 0) && !tu_admit_more(true, 1);
        put("lone writer and admission charging", ok && adm, std::to_string(ok) + std::to_string(adm));
    }
    {
        // the reader to follow the current turn: two long readers rotate (the other one); a lone reader follows
        // itself; a short reader follows itself ahead of a long one; a reader not begun is skipped; none: -1
        std::vector<tu_reader> r(2);
        r[0] = {0, 0, 30000, 32768, 5, 20, true, true};
        r[1] = {1, 0, 30000, 32768, 4, 21, false, true};
        const int a = tu_follow(r, 0, fast);
        const int b = tu_follow(std::vector<tu_reader>{r[0]}, 0, fast);
        std::vector<tu_reader> s2(2);
        s2[0] = {0, 0, 2000, 3000, 5, 20, true, true};
        s2[1] = {1, 0, 60000, 60000, 4, 21, false, true};
        const int c = tu_follow(s2, 0, fast);
        r[1].begun = false;
        const int d = tu_follow(r, 0, fast);
        r[0].begun = false;
        const int e = tu_follow(r, 0, fast);
        // no turn in progress: the order's first begun reader
        r[0].begun = r[1].begun = true;
        const int f = tu_follow(r, -1, fast);
        put("the reader to follow the current turn", a == 1 && b == 0 && c == 0 && d == 0 && e == -1 && f == 1,
            std::to_string(a) + " " + std::to_string(b) + " " + std::to_string(c) + " " + std::to_string(d) + " " +
                std::to_string(e) + " " + std::to_string(f));
    }

    std::printf("fnfl-splice-test: %s\n", n_fail == 0 ? "all PASS" : "FAIL");
    return n_fail == 0 ? 0 : 1;
}
