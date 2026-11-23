#include "fn/tools/think_test.h"

#include "fn/runtime/think_budget.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace fn::tools {

namespace {

using fn::think::config;
using fn::think::tracker;

constexpr int32_t tok_open = 7, tok_eos = 9;

// a scripted model: the next token from the turn's generated tokens alone. Reasoning filler tokens are 1000 + .., the
// answer's 2000 + ..; it closes its reasoning after n_reason tokens (n_reason2 in a reopened block), may start a false
// prefix of a multi-token close at false_at, answers n_content tokens, may reopen reasoning at content index reopen_at,
// and ends with tok_eos. A forced close is simply part of its history. alt (C2): the model ends its reasoning with
// this other ending instead (Qwen's <tool_call>), which its parser accepts too
struct script {
    bool open_at_start = true;
    bool opens_itself = false; // raw completion: it opens reasoning itself first
    int32_t n_reason = 20;
    int32_t false_at = -1;
    int32_t n_content = 6;
    int32_t reopen_at = -1;
    int32_t n_reason2 = 8;
    std::vector<int32_t> alt;
};

int32_t next_of(const script & s, const std::vector<int32_t> & close, const std::vector<int32_t> & g) {
    bool in = s.open_at_start, falsed = false, reopened = false;
    int32_t blocks = in ? 1 : 0, r = 0, m = 0, ma = 0, c = 0, false_start = -1;
    const auto la = static_cast<int32_t>(s.alt.size());
    for (const int32_t t : g) {
        if (!in) {
            if (t == tok_open) {
                in = true;
                reopened = reopened || blocks > 0;
                ++blocks;
                r = 0;
                m = 0;
                continue;
            }
            ++c;
            continue;
        }
        const bool cm = t == close[static_cast<size_t>(m)];
        const bool am = la > 0 && t == s.alt[static_cast<size_t>(ma)];
        if (cm || am) {
            false_start = m == 0 && ma == 0 ? r : false_start;
            m = cm ? m + 1 : 0;
            ma = am ? ma + 1 : 0;
            if (m == static_cast<int32_t>(close.size()) || (la > 0 && ma == la)) {
                in = false;
                m = 0;
                ma = 0;
                c = 0;
            }
            continue;
        }
        falsed = falsed || m > 0 || ma > 0;
        m = 0;
        ma = 0;
        ++r;
    }
    if (!in) {
        if (g.empty() && s.opens_itself) {
            return tok_open;
        }
        if (s.reopen_at >= 0 && !reopened && blocks > 0 && c == s.reopen_at) {
            return tok_open;
        }
        return c >= s.n_content ? tok_eos : 2000 + static_cast<int32_t>(g.size() % 31);
    }
    const int32_t limit = blocks <= 1 ? s.n_reason : s.n_reason2;
    if (ma > 0 && r >= limit) {
        return s.alt[static_cast<size_t>(ma)]; // its own ending, under way
    }
    if (m > 0) {
        // its own false prefix breaks off; any other prefix (its close, or a forced one) it completes
        const bool own_false = s.false_at >= 0 && !falsed && false_start == s.false_at;
        return own_false ? 1000 + static_cast<int32_t>(g.size() % 37) : close[static_cast<size_t>(m)];
    }
    if (s.false_at >= 0 && !falsed && r == s.false_at && blocks <= 1) {
        return close[0];
    }
    if (r >= limit) {
        return la > 0 ? s.alt[0] : close[0];
    }
    return 1000 + static_cast<int32_t>(g.size() % 37);
}

// the model's next token from the turn's generated tokens
using model = std::function<int32_t(const std::vector<int32_t> &)>;

model of(const script & s, const config & c) {
    return [&s, &c](const std::vector<int32_t> & g) { return next_of(s, c.close, g); };
}

struct run_out {
    std::vector<int32_t> g;
    int32_t counted = 0, forced = 0, added = 0, extra = 0;
    int32_t eos = tok_eos; // the model's end token
    bool reasoning = false;
    int32_t bad = 0; // a free position the tracker said was forced, or a forced one not taken
};

bool done(const run_out & o, int32_t limit) {
    return (!o.g.empty() && o.g.back() == o.eos) || static_cast<int32_t>(o.g.size()) >= limit;
}

void emit(tracker & tr, run_out & o, int32_t t, bool free_pos) {
    if (free_pos && tr.forced() >= 0) {
        ++o.bad;
    }
    o.added += tr.observe(t);
    o.g.push_back(t);
}

run_out finish(const tracker & tr, run_out o) {
    o.counted = tr.counted();
    o.forced = tr.forced_n();
    o.reasoning = tr.reasoning();
    o.extra = tr.extra_n();
    return o;
}

// token by token: the forced token, else the model's
run_out scalar(const model & m, const config & c, int32_t limit, int32_t eos = tok_eos) {
    tracker tr;
    tr.reset(c);
    run_out o;
    o.eos = eos;
    while (!done(o, limit)) {
        const int32_t f = tr.forced();
        emit(tr, o, f >= 0 ? f : m(o.g), false);
    }
    return finish(tr, o);
}

run_out scalar(const script & s, const config & c, int32_t limit) {
    return scalar(of(s, c), c, limit);
}

// one speculative step as the engines run it: d proposals (the model's continuation, some corrupted, with close and
// open tokens among the corruptions), the window cut at the tracker's cap and the room, a forced step alone, else
// greedy prefix acceptance and the model's token after the accepted ones; more: further corruptions (UTF-8 pieces)
void spec_step(const model & m, const config & c, tracker & tr, run_out & o, int32_t limit, int32_t d,
               std::mt19937 & rng, const std::vector<int32_t> & more = {}) {
    const int32_t f = tr.forced();
    if (f >= 0) {
        if (tr.cap(nullptr, 0) != 1) {
            ++o.bad;
        }
        emit(tr, o, f, false);
        return;
    }
    // FNFL's step budget: inside reasoning no more than the tracker's room (each position before it must be free)
    const int32_t tb = std::min(limit - static_cast<int32_t>(o.g.size()), tr.room());
    if (tb < 1) {
        ++o.bad;
    }
    if (tb <= 1) {
        emit(tr, o, m(o.g), true);
        return;
    }
    d = std::min(d, tb - 1);
    std::vector<int32_t> props, h = o.g;
    std::vector<int32_t> alphabet = {c.close[0], c.close.back(), tok_open, 1001, 2002};
    alphabet.insert(alphabet.end(), more.begin(), more.end());
    for (int32_t j = 0; j < d; ++j) {
        int32_t t = m(h);
        if (rng() % 4 == 0) {
            t = alphabet[rng() % alphabet.size()];
        }
        props.push_back(t);
        h.push_back(t);
    }
    const int32_t room = limit - static_cast<int32_t>(o.g.size());
    const int32_t cap = std::min(tr.cap(props.data(), d), room);
    const int32_t keep = cap - 1;
    for (int32_t j = 0; j < keep; ++j) {
        const int32_t want = m(o.g);
        if (props[static_cast<size_t>(j)] != want) {
            break;
        }
        emit(tr, o, want, true);
        if (want == o.eos) {
            return;
        }
    }
    emit(tr, o, m(o.g), true);
}

void spec_step(const script & s, const config & c, tracker & tr, run_out & o, int32_t limit, int32_t d,
               std::mt19937 & rng) {
    spec_step(of(s, c), c, tr, o, limit, d, rng);
}

run_out spec(const model & m, const config & c, int32_t limit, int32_t d, uint32_t seed,
             const std::vector<int32_t> & more = {}, int32_t eos = tok_eos) {
    tracker tr;
    tr.reset(c);
    run_out o;
    o.eos = eos;
    std::mt19937 rng(seed);
    while (!done(o, limit)) {
        spec_step(m, c, tr, o, limit, d, rng, more);
    }
    return finish(tr, o);
}

// IFR1 R3: a listed model: its reasoning tokens in order (the prompt opened reasoning), its own close when they run
// out, then its answer and eos. One-token endings only (each engine's model); a forced close is part of its history
struct listed {
    std::vector<int32_t> reason, answer;
    int32_t eos = tok_eos;
};

model of(const listed & l, const config & c) {
    return [&l, &c](const std::vector<int32_t> & g) {
        const auto ending = [&](int32_t t) {
            bool e = t == c.close.back();
            for (const std::vector<int32_t> & x : c.ends) {
                e = e || x.back() == t;
            }
            return e;
        };
        size_t r = 0, a = 0;
        bool in = true;
        for (const int32_t t : g) {
            if (!in) {
                ++a;
            } else if (ending(t)) {
                in = false;
            } else {
                ++r;
            }
        }
        if (in) {
            return r < l.reason.size() ? l.reason[r] : c.close[0];
        }
        return a < l.answer.size() ? l.answer[a] : l.eos;
    };
}

// IFR1 R3: a fixture an engine's chat check wrote with the model's own tokenizer (server-reasoning-check
// reasoning_check_utf8): the template's sequences, the server's UTF-8 table and listed cases, each with the tokens the
// rule gives (want, from the bytes) and the tokens the tracker without a table gives (base)
struct fixture_case {
    std::string name;
    int32_t budget = 0, limit = 0;
    listed l;
    run_out want, base;
};

struct fixture {
    std::string model;
    config c;
    std::vector<fixture_case> cases;
};

bool read_fixture(const std::string & path, fixture & f, std::string & err) {
    std::ifstream in(path);
    std::string w;
    const auto word = [&](const char * k) {
        if (!(in >> w) || w != k) {
            err = path + ": expected \"" + k + "\", got \"" + w + "\"";
            return false;
        }
        return true;
    };
    const auto list = [&](std::vector<int32_t> & v) {
        int64_t n = 0;
        if (!(in >> n) || n < 0 || n > (1 << 22)) {
            err = path + ": a bad list length";
            return false;
        }
        v.assign(static_cast<size_t>(n), 0);
        for (int32_t & x : v) {
            if (!(in >> x)) {
                err = path + ": a list cut short";
                return false;
            }
        }
        return true;
    };
    const auto outcome = [&](run_out & o, int32_t eos) {
        int32_t r = 0;
        o.eos = eos;
        if (!list(o.g) || !(in >> o.counted >> o.forced >> o.extra >> r)) {
            return false;
        }
        o.reasoning = r != 0;
        return true;
    };
    if (!in || !word("think-utf8") || !(in >> f.model) || !word("open") || !list(f.c.open) || !word("close") ||
        !list(f.c.close) || !word("ends")) {
        return err.empty() ? (err = path + ": unreadable", false) : false;
    }
    int32_t k = 0;
    in >> k;
    f.c.ends.assign(static_cast<size_t>(std::max(k, 0)), {});
    for (std::vector<int32_t> & e : f.c.ends) {
        if (!list(e)) {
            return false;
        }
    }
    int32_t eos = 0;
    std::vector<int32_t> table;
    if (!word("eos") || !(in >> eos) || !word("table") || !list(table) || !word("cases") || !(in >> k)) {
        return err.empty() ? (err = path + ": a bad header", false) : false;
    }
    auto t = std::make_shared<std::vector<uint32_t>>();
    for (const int32_t x : table) {
        t->push_back(static_cast<uint32_t>(x));
    }
    f.c.utf8 = t;
    f.c.open_at_start = true;
    for (int32_t i = 0; i < k; ++i) {
        fixture_case fc;
        fc.l.eos = eos;
        if (!word("case") || !(in >> fc.name >> fc.budget >> fc.limit) || !word("reason") || !list(fc.l.reason) ||
            !word("answer") || !list(fc.l.answer) || !word("want") || !outcome(fc.want, eos) || !word("base") ||
            !outcome(fc.base, eos)) {
            return err.empty() ? (err = path + ": case " + std::to_string(i) + " unreadable", false) : false;
        }
        f.cases.push_back(fc);
    }
    return true;
}

std::string str(const std::vector<int32_t> & g) {
    std::string t;
    for (size_t i = 0; i < g.size() && i < 40; ++i) {
        t += (i ? " " : "") + std::to_string(g[i]);
    }
    return t + (g.size() > 40 ? " ..." : "");
}

} // namespace

int think_test(const std::vector<std::string> & fixtures) {
    int fails = 0;
    const auto report = [&](bool ok, const std::string & line) {
        std::printf("think: %s %s\n", ok ? "PASS" : "FAIL", line.c_str());
        fails += ok ? 0 : 1;
    };
    const auto same_spec_m = [&](const model & m, const config & c, int32_t limit, const run_out & want,
                                 int32_t & runs, const std::vector<int32_t> & more) {
        bool ok = true;
        for (int32_t d = 1; d <= 4; ++d) {
            for (uint32_t seed = 1; seed <= 25; ++seed) {
                const run_out o = spec(m, c, limit, d, seed * 7919u + static_cast<uint32_t>(d), more, want.eos);
                ok = ok && o.g == want.g && o.counted == want.counted && o.forced == want.forced && o.bad == 0 &&
                     o.added == want.added && o.extra == want.extra;
                ++runs;
            }
        }
        return ok;
    };
    const auto same_spec = [&](const script & s, const config & c, int32_t limit, const run_out & want,
                               int32_t & runs) {
        return same_spec_m(of(s, c), c, limit, want, runs, {});
    };
    const std::vector<int32_t> one = {5}, two = {5, 6};
    const auto cfg = [](int32_t b, const std::vector<int32_t> & close, bool open) {
        config c;
        c.budget = b;
        c.open = {tok_open};
        c.close = close;
        c.open_at_start = open;
        return c;
    };

    // the configuration's checks
    {
        std::string err;
        config c = cfg(-2, one, true);
        const bool below = !fn::think::check(c, err);
        c = cfg(10, {}, true);
        const bool no_close = !fn::think::check(c, err);
        c = cfg(-1, {}, true);
        const bool unlimited = fn::think::check(c, err);
        report(below && no_close && unlimited,
               "check: a budget of -2 and a budget without a closing sequence refused, unlimited with none taken");
    }
    // unlimited: nothing forced, the window and the tokens unchanged; FBH1 A3: with the closing sequence named the
    // reasoning is counted all the same (the usage report's), without it nothing is tracked
    {
        tracker tr;
        tr.reset(cfg(-1, one, true));
        const int32_t props[4] = {1000, 5, 2000, 2001};
        const bool free_run = !tr.active() && tr.tracks() && tr.forced() < 0 && tr.cap(props, 4) == 5 &&
                              tr.room() == INT32_MAX && tr.observe(1000) == 1;
        tracker none;
        none.reset(cfg(-1, {}, true));
        const bool idle = !none.active() && !none.tracks() && none.observe(1000) == 0 && none.counted() == 0;
        script s;
        const run_out o = scalar(s, cfg(-1, one, true), 200);
        int32_t runs = 0;
        const bool spec_ok = same_spec(s, cfg(-1, one, true), 200, o, runs);
        // a message without a budget is never fed
        config cm = cfg(-1, one, true);
        cm.message = {40, 41, 42};
        const run_out om = scalar(s, cm, 200);
        script s2;
        s2.n_reason = 6;
        s2.reopen_at = 2;
        const run_out o2 = scalar(s2, cfg(-1, one, true), 400);
        report(free_run && idle && o.g.size() == 28 && o.g[20] == 5 && o.counted == 20 && o.added == 20 &&
                   o.forced == 0 && o.extra == 0 && spec_ok && om.g == o.g && om.counted == 20 && o2.counted == 14 &&
                   o2.forced == 0,
               "unlimited: nothing forced, cap = proposals + 1, the natural close at 20 and its 20 reasoning tokens "
               "counted (6 + 8 over a reopened block; nothing without the sequences; a message never fed), " +
                   std::to_string(runs) + " speculative runs equal: " + str(o.g));
    }
    // every budget 0 .. 25 around a natural close at 20, single-token close
    {
        script s;
        bool ok = true;
        int32_t runs = 0;
        std::string first_bad;
        for (int32_t b = 0; b <= 25; ++b) {
            const config c = cfg(b, one, true);
            const run_out o = scalar(s, c, 400);
            const int32_t at = std::min(b, 20);
            const bool shape = o.counted == at && o.g[static_cast<size_t>(at)] == 5 && o.forced == (b <= 20 ? 1 : 0) &&
                               o.added == at && !o.reasoning && o.g.back() == tok_eos &&
                               static_cast<int32_t>(o.g.size()) == at + 1 + 6 + 1;
            const bool sp = same_spec(s, c, 400, o, runs);
            if ((!shape || !sp) && first_bad.empty()) {
                first_bad = " (first failure at budget " + std::to_string(b) + ": " + str(o.g) + ")";
            }
            ok = ok && shape && sp;
        }
        report(ok,
               "budgets 0 .. 25, close at min(budget, 20) (forced up to 20: at 20 the decision is the budget's), the "
               "answer after it, " + std::to_string(runs) + " speculative runs (d 1 .. 4) equal to token by token" +
                   first_bad);
    }
    // a draft crossing the boundary: the window ends at the budget, past a natural close it does not
    {
        tracker tr;
        tr.reset(cfg(10, one, true));
        for (int32_t i = 0; i < 8; ++i) {
            tr.observe(1000 + i);
        }
        const int32_t fill[4] = {1100, 1101, 1102, 1103}, closing[4] = {5, 2000, 2001, 2002};
        const int32_t c1 = tr.cap(fill, 4), c2 = tr.cap(closing, 4);
        tr.observe(1100);
        tr.observe(1101);
        const int32_t f = tr.forced(), c3 = tr.cap(fill, 4);
        report(c1 == 2 && c2 == 5 && f == 5 && c3 == 1,
               "window at 8 of 10: 2 tokens with filler proposals (" + std::to_string(c1) + "), 5 when the first "
               "closes (" + std::to_string(c2) + "); at 10 the close forced (" + std::to_string(f) + ", window " +
               std::to_string(c3) + ")");
    }
    // multi-token close with the model's own false prefix at 8 (counted once it breaks off)
    {
        script s;
        s.false_at = 8;
        bool ok = true;
        int32_t runs = 0;
        std::string first_bad;
        for (int32_t b = 0; b <= 25; ++b) {
            const config c = cfg(b, two, true);
            const run_out o = scalar(s, c, 400);
            // the reasoning: 20 filler tokens and the false prefix's one token (21 counted naturally); at budget 9
            // the false prefix at 8 counted tokens is completed into the close instead (8 counted)
            const int32_t at = b == 9 ? 8 : std::min(b, 21);
            bool shape = o.counted == at && o.added == at && !o.reasoning && o.g.back() == tok_eos;
            // the close sequence ends the reasoning block, right after `at` counted tokens
            const size_t end = static_cast<size_t>(at) + 2;
            shape = shape && o.g.size() > end && o.g[end - 2] == 5 && o.g[end - 1] == 6;
            const bool sp = same_spec(s, c, 400, o, runs);
            if ((!shape || !sp) && first_bad.empty()) {
                first_bad = " (first failure at budget " + std::to_string(b) + ": " + str(o.g) + ")";
            }
            ok = ok && shape && sp;
        }
        report(ok, "two-token close, a false prefix at 8: budgets 0 .. 25 end the reasoning with 5 6 after min(budget, "
                   "21) counted tokens, " + std::to_string(runs) + " speculative runs equal" + first_bad);
        // at budget 9 the false prefix (5 at 8 counted tokens) becomes the close: 6 forced, not 5 again
        const run_out o = scalar(s, cfg(9, two, true), 400);
        report(o.counted == 8 && o.forced == 1 && o.g[8] == 5 && o.g[9] == 6,
               "budget 9: the model's 5 at 8 completed with a forced 6: " + str(o.g));
    }
    // zero budget: the first token is the close
    {
        script s;
        const run_out o = scalar(s, cfg(0, two, true), 400);
        int32_t runs = 0;
        const bool sp = same_spec(s, cfg(0, two, true), 400, o, runs);
        report(o.g.size() > 2 && o.g[0] == 5 && o.g[1] == 6 && o.counted == 0 && o.forced == 2 && sp,
               "budget 0: the first tokens are the close 5 6: " + str(o.g));
    }
    // no thinking: the prompt closed reasoning (or never opened it): nothing counted or forced
    {
        script s;
        s.open_at_start = false;
        const run_out o = scalar(s, cfg(0, one, false), 400);
        report(o.counted == 0 && o.forced == 0 && o.g.size() == 7 && o.g[0] == 2000,
               "no thinking at budget 0: the answer untouched: " + str(o.g));
    }
    // raw completion: the model opens reasoning itself; the budget counts from its open
    {
        script s;
        s.open_at_start = false;
        s.opens_itself = true;
        const run_out o = scalar(s, cfg(3, one, false), 400);
        int32_t runs = 0;
        const bool sp = same_spec(s, cfg(3, one, false), 400, o, runs);
        report(o.g[0] == tok_open && o.g[4] == 5 && o.counted == 3 && o.forced == 1 && sp,
               "raw completion, budget 3: open, 3 reasoning tokens, the forced close: " + str(o.g));
    }
    // reopened reasoning shares the turn's budget
    {
        script s;
        s.n_reason = 6;
        s.reopen_at = 2;
        s.n_reason2 = 8;
        const run_out o = scalar(s, cfg(10, one, true), 400);
        int32_t runs = 0;
        const bool sp = same_spec(s, cfg(10, one, true), 400, o, runs);
        // 6 reasoning, close, 2 answer, open, 4 reasoning (the budget's rest), forced close
        report(o.counted == 10 && o.forced == 1 && o.g[6] == 5 && o.g[9] == tok_open && o.g[14] == 5 && sp,
               "reopened at budget 10: 6 + 4 counted, the second close forced: " + str(o.g));
    }
    // the total limit ends the request inside the close: the reasoning stays open, the limit is not extended
    {
        script s;
        const run_out o = scalar(s, cfg(4, two, true), 5);
        int32_t runs = 0;
        const bool sp = same_spec(s, cfg(4, two, true), 5, o, runs);
        report(o.g.size() == 5 && o.g[4] == 5 && o.reasoning && o.counted == 4 && sp,
               "a total limit of 5 at budget 4 with a two-token close: 4 reasoning and 5, the reasoning unclosed: " +
                   str(o.g));
    }
    // two slots with unequal budgets stepped in turn: each equals its lone run (no state shared)
    {
        script s;
        const config ca = cfg(3, one, true), cb = cfg(12, two, true);
        const run_out wa = scalar(s, ca, 400), wb = scalar(s, cb, 400);
        tracker ta, tb;
        ta.reset(ca);
        tb.reset(cb);
        run_out oa, ob;
        std::mt19937 ra(11), rb(12);
        while (!done(oa, 400) || !done(ob, 400)) {
            if (!done(oa, 400)) {
                spec_step(s, ca, ta, oa, 400, 4, ra);
            }
            if (!done(ob, 400)) {
                spec_step(s, cb, tb, ob, 400, 3, rb);
            }
        }
        oa = finish(ta, oa);
        ob = finish(tb, ob);
        report(oa.g == wa.g && ob.g == wb.g && oa.counted == 3 && ob.counted == 12 && oa.bad == 0 && ob.bad == 0,
               "two slots, budgets 3 and 12, stepped in turn: each equal to its lone run");
    }
    // a cancelled request's slot reused: the next request's state starts fresh
    {
        script s;
        tracker tr;
        tr.reset(cfg(2, one, true));
        run_out o;
        std::mt19937 r(5);
        spec_step(s, cfg(2, one, true), tr, o, 400, 4, r);
        spec_step(s, cfg(2, one, true), tr, o, 400, 4, r);
        tr.reset(cfg(7, one, true));
        run_out o2;
        while (!done(o2, 400)) {
            spec_step(s, cfg(7, one, true), tr, o2, 400, 4, r);
        }
        o2 = finish(tr, o2);
        const run_out want = scalar(s, cfg(7, one, true), 400);
        report(o2.g == want.g && o2.counted == 7, "a slot reused after a cancel at budget 2: budget 7's run is fresh");
    }
    // C2: an ending the parser accepts besides the close (Qwen's <tool_call>, here 8) ends the reasoning naturally;
    // only the close (5) is ever forced
    {
        script s;
        s.alt = {8};
        bool ok = true;
        int32_t runs = 0;
        std::string first_bad;
        for (int32_t b = 0; b <= 25; ++b) {
            config c = cfg(b, one, true);
            c.ends = {{8}};
            const run_out o = scalar(s, c, 400);
            const int32_t at = std::min(b, 20);
            const bool no_5 = std::find(o.g.begin(), o.g.end(), 5) == o.g.end();
            const bool shape = o.counted == at && !o.reasoning && o.g.back() == tok_eos &&
                               (b <= 20 ? o.g[static_cast<size_t>(b)] == 5 && o.forced == 1
                                        : o.g[20] == 8 && o.forced == 0 && no_5);
            const bool sp = same_spec(s, c, 400, o, runs);
            if ((!shape || !sp) && first_bad.empty()) {
                first_bad = " (first failure at budget " + std::to_string(b) + ": " + str(o.g) + ")";
            }
            ok = ok && shape && sp;
        }
        report(ok, "another ending (8) at 20: budgets 0 .. 20 force the close 5 (at 20 the decision is the budget's), "
                   "21 .. 25 end at 8 with nothing forced and 20 counted, " +
                   std::to_string(runs) + " speculative runs equal" + first_bad);
        // the gap C2 closes: without the ending configured, the content after 8 is counted and 5 forced into it
        const run_out o = scalar(s, cfg(25, one, true), 400);
        const bool five = std::find(o.g.begin() + 21, o.g.end(), 5) != o.g.end();
        report(five && o.counted == 25 && o.forced == 1,
               "without the ending: the answer after 8 counted to 25 and 5 forced into it: " + str(o.g));
    }
    // C2 with a two-token close and a two-token ending sharing its first token: 5 6 forced, 5 8 the model's own
    {
        script s;
        s.alt = {5, 8};
        bool ok = true;
        int32_t runs = 0;
        std::string first_bad;
        for (int32_t b = 0; b <= 25; ++b) {
            config c = cfg(b, two, true);
            c.ends = {{5, 8}};
            const run_out o = scalar(s, c, 400);
            const int32_t at = std::min(b, 20);
            const bool shape = o.counted == at && !o.reasoning && o.g.back() == tok_eos &&
                               o.g[static_cast<size_t>(at)] == 5 &&
                               (b <= 20   ? o.g[static_cast<size_t>(at) + 1] == 6 && o.forced == 2
                                : b == 21 ? o.g[21] == 6 && o.forced == 1
                                          : o.g[21] == 8 && o.forced == 0);
            const bool sp = same_spec(s, c, 400, o, runs);
            if ((!shape || !sp) && first_bad.empty()) {
                first_bad = " (first failure at budget " + std::to_string(b) + ": " + str(o.g) + ")";
            }
            ok = ok && shape && sp;
        }
        // at 21 the model's own 5 waits as the 21st token, so the budget completes the close with 6
        report(ok, "a two-token ending 5 8 beside the close 5 6: budgets 0 .. 20 force 5 6, 21 completes the model's 5 "
                   "with a forced 6, 22 .. 25 end at 5 8 with nothing forced, " +
                   std::to_string(runs) + " speculative runs equal" + first_bad);
    }
    // FBH1 A1: the budget message (40 41 42) fed before the forced close and reserved inside the budget
    const std::vector<int32_t> msg = {40, 41, 42};
    const auto with_msg = [&](config c) {
        c.message = msg;
        return c;
    };
    // the reserve: budgets 12 .. 30 (>= 4 x 3) around a natural close at 20: the free reasoning stops at budget - 3,
    // then the message and the close; the natural close first from budget 24
    {
        script s;
        bool ok = true;
        int32_t runs = 0;
        std::string first_bad;
        for (int32_t b = 12; b <= 30; ++b) {
            const config c = with_msg(cfg(b, one, true));
            const run_out o = scalar(s, c, 400);
            const int32_t free_n = std::min(b - 3, 20);
            const auto at = static_cast<size_t>(free_n);
            bool shape = !o.reasoning && o.g.back() == tok_eos && o.extra == 0;
            if (b - 3 <= 20) {
                // at 20 the decision is the budget's: the message, not the model's own close
                shape = shape && o.g.size() == at + 3 + 1 + 6 + 1 && o.g[at] == 40 && o.g[at + 1] == 41 &&
                        o.g[at + 2] == 42 && o.g[at + 3] == 5 && o.counted == b && o.added == b && o.forced == 4;
            } else {
                shape = shape && o.g.size() == 20 + 1 + 6 + 1 && o.g[20] == 5 && o.counted == 20 && o.forced == 0 &&
                        std::find(o.g.begin(), o.g.end(), 40) == o.g.end();
            }
            int32_t r = 0;
            const bool sp = same_spec_m(of(s, c), c, 400, o, r, msg);
            runs += r;
            if ((!shape || !sp) && first_bad.empty()) {
                first_bad = " (first failure at budget " + std::to_string(b) + ": " + str(o.g) + ")";
            }
            ok = ok && shape && sp;
        }
        report(ok, "message 40 41 42, budgets 12 .. 30: budget - 3 free tokens, the message, the close, counted = the "
                   "budget (4 forced); from 24 the natural close at 20 and no message, " + std::to_string(runs) +
                       " speculative runs equal" + first_bad);
    }
    // tiny budgets (below 4 x 3) and budget 0: no message, the same tokens as without one
    {
        script s;
        bool ok = true;
        int32_t runs = 0;
        for (int32_t b = 0; b <= 11; ++b) {
            const config c = with_msg(cfg(b, two, true)), c0 = cfg(b, two, true);
            const run_out o = scalar(s, c, 400), w = scalar(s, c0, 400);
            int32_t r = 0;
            ok = ok && o.g == w.g && o.counted == w.counted && o.forced == w.forced && o.counted == b &&
                 std::find(o.g.begin(), o.g.end(), 40) == o.g.end() && same_spec_m(of(s, c), c, 400, o, r, msg);
            runs += r;
        }
        report(ok && !fn::think::reserve_ok(11, 3) && fn::think::reserve_ok(12, 3) && !fn::think::reserve_ok(0, 0),
               "message, budgets 0 .. 11 (below 4 x its 3 tokens): no message, the close alone as without one, " +
                   std::to_string(runs) + " speculative runs equal");
    }
    // the window and the room at the reserve: 8 of budget 13's 10 free tokens, then each message token alone
    {
        tracker tr;
        tr.reset(with_msg(cfg(13, one, true)));
        for (int32_t i = 0; i < 8; ++i) {
            tr.observe(1000 + i);
        }
        const int32_t fill[4] = {1100, 1101, 1102, 1103};
        const int32_t c1 = tr.cap(fill, 4), r1 = tr.room();
        tr.observe(1100);
        tr.observe(1101);
        bool forced = tr.forced() == 40 && tr.cap(fill, 4) == 1 && tr.room() == 0 && tr.counted() == 10;
        for (const int32_t t : {40, 41, 42}) {
            forced = forced && tr.forced() == t && tr.cap(fill, 4) == 1 && tr.observe(t) == 1;
        }
        forced = forced && tr.forced() == 5 && tr.counted() == 13 && tr.forced_n() == 3;
        tr.observe(5);
        report(c1 == 2 && r1 == 2 && forced && !tr.reasoning() && tr.forced_n() == 4,
               "message at budget 13: the window at 8 of 10 free tokens is 2 (" + std::to_string(c1) + "), then 40 41 "
               "42 and 5 forced one a step, 13 counted");
    }
    // a false prefix of the message: the model's own 40 41 just before the reserve are reasoning like any other; the
    // message is fed whole after them
    {
        listed l;
        for (int32_t i = 0; i < 8; ++i) {
            l.reason.push_back(1000 + i);
        }
        l.reason.insert(l.reason.end(), {40, 41, 1008, 1009, 1010});
        l.answer = {2000, 2001};
        const config c = with_msg(cfg(13, one, true));
        const run_out o = scalar(of(l, c), c, 400);
        int32_t runs = 0;
        const bool sp = same_spec_m(of(l, c), c, 400, o, runs, msg);
        const std::vector<int32_t> want = {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 40, 41, 40, 41, 42, 5,
                                           2000, 2001, tok_eos};
        report(o.g == want && o.counted == 13 && o.forced == 4 && sp,
               "message, the model's own 40 41 at 8 and 9 of 10 free tokens: counted as reasoning, the message whole "
               "after them, " + std::to_string(runs) + " speculative runs equal: " + str(o.g));
    }
    // message tokens that begin another ending (41 8) wait and are counted when the message goes on; the model's own
    // first closing token standing at the reserve is completed into the close, without the message
    {
        script s;
        config c = with_msg(cfg(13, one, true));
        c.ends = {{41, 8}};
        const run_out o = scalar(s, c, 400);
        int32_t runs = 0;
        const bool sp = same_spec_m(of(s, c), c, 400, o, runs, msg);
        const bool waits = o.counted == 13 && o.forced == 4 && o.g[10] == 40 && o.g[11] == 41 && o.g[12] == 42 &&
                           o.g[13] == 5 && !o.reasoning;
        script f;
        f.false_at = 9;
        const config c2 = with_msg(cfg(13, two, true));
        const run_out o2 = scalar(f, c2, 400);
        const bool sp2 = same_spec_m(of(f, c2), c2, 400, o2, runs, msg);
        const bool own = o2.counted == 9 && o2.forced == 1 && o2.g[9] == 5 && o2.g[10] == 6 && !o2.reasoning &&
                         std::find(o2.g.begin(), o2.g.end(), 40) == o2.g.end();
        report(waits && sp && own && sp2,
               "message beside an ending 41 8: its 41 waits, then counted (13); the model's own 5 at the reserve of a "
               "5 6 close completed with a forced 6 and no message (9 counted), " + std::to_string(runs) +
                   " speculative runs equal: " + str(o.g) + " | " + str(o2.g));
    }
    // the total limit inside the message leaves the reasoning unclosed; reasoning reopened after a fed message is
    // closed at once (the budget is spent), the message not fed again
    {
        script s;
        const config c = with_msg(cfg(13, one, true));
        const run_out o = scalar(s, c, 12);
        int32_t runs = 0;
        const bool sp = same_spec_m(of(s, c), c, 12, o, runs, msg);
        const bool cut = o.g.size() == 12 && o.g[10] == 40 && o.g[11] == 41 && o.reasoning && o.counted == 12;
        script s2;
        s2.reopen_at = 2;
        const run_out o2 = scalar(s2, c, 400);
        const bool sp2 = same_spec_m(of(s2, c), c, 400, o2, runs, msg);
        // 10 free, 40 41 42, 5, 2 answer tokens, the open, 5 forced at once
        const bool again = o2.counted == 13 && o2.forced == 5 && o2.g[16] == tok_open && o2.g[17] == 5 &&
                           std::count(o2.g.begin(), o2.g.end(), 40) == 1 && !o2.reasoning;
        report(cut && sp && again && sp2,
               "message: a total limit of 12 ends inside it, the reasoning unclosed; reopened after it, the close at "
               "once and no second message, " + std::to_string(runs) + " speculative runs equal: " + str(o2.g));
    }
    // the message's checks
    {
        std::string err;
        config c = with_msg(cfg(13, two, true));
        const bool taken = fn::think::check(c, err);
        c.message = {40, 5, 6, 41};
        const bool holds_close = !fn::think::check(c, err);
        c.message = {40, 41, 5};
        const bool ends_in_prefix = !fn::think::check(c, err);
        c.message = {40, 8, 41};
        c.ends = {{8}};
        const bool holds_end = !fn::think::check(c, err);
        c.ends.clear();
        c.message.assign(static_cast<size_t>(fn::think::max_message) + 1, 40);
        const bool too_long = !fn::think::check(c, err);
        c.message.assign(static_cast<size_t>(fn::think::max_message), 40);
        report(taken && holds_close && ends_in_prefix && holds_end && too_long && fn::think::check(c, err),
               "check: a message holding the close or another ending, ending with the close's first token, or longer "
               "than 256 tokens refused; 256 tokens taken");
    }
    // FBH1 K1: the forced close's form: the lead 60 (the model's newline), the close, the tail 61 (its blank line).
    // The listed model closes the same way on its own: 19 reasoning tokens and its newline 60, the close, then its
    // answer 61 2000 2001
    const auto formed = [](config c) {
        c.lead = {60};
        c.tail = {61};
        return c;
    };
    listed own;
    for (int32_t i = 0; i < 19; ++i) {
        own.reason.push_back(1000 + i);
    }
    own.reason.push_back(60);
    own.answer = {61, 2000, 2001};
    const std::vector<int32_t> k1_more = {60, 61, 62, 40, 41, 42};
    const auto fillers = [](int32_t n) {
        std::vector<int32_t> v;
        for (int32_t i = 0; i < n; ++i) {
            v.push_back(1000 + i);
        }
        return v;
    };
    const auto then = [](std::vector<int32_t> v, const std::vector<int32_t> & more) {
        v.insert(v.end(), more.begin(), more.end());
        return v;
    };
    // every budget 0 .. 25 without a message
    {
        bool ok = true;
        int32_t runs = 0;
        std::string first_bad;
        for (int32_t b = 0; b <= 25; ++b) {
            const config c = formed(cfg(b, one, true));
            const run_out o = scalar(of(own, c), c, 400);
            const std::vector<int32_t> want =
                then(fillers(std::min(std::max(b - 1, 0), 19)), {60, 5, 61, 2000, 2001, tok_eos});
            const int32_t forced = b <= 20 ? 3 : b == 21 ? 2 : 0;
            const bool shape = o.g == want && o.counted == std::min(b, 20) && o.added == o.counted && o.extra == 0 &&
                               o.forced == forced && !o.reasoning;
            int32_t r = 0;
            const bool sp = same_spec_m(of(own, c), c, 400, o, r, k1_more);
            runs += r;
            if ((!shape || !sp) && first_bad.empty()) {
                first_bad = " (first failure at budget " + std::to_string(b) + ": " + str(o.g) + ")";
            }
            ok = ok && shape && sp;
        }
        report(ok, "K1 lead 60 and tail 61, budgets 0 .. 25 without a message: budget - 1 free tokens, then 60 5 61 "
                   "forced, counted = the budget (budget 0: 60 fed and not counted, 0 counted); at 21 the model's own "
                   "60 is the last free token and only 5 61 are forced (20 counted); from 22 the model's own close and "
                   "blank line, nothing forced; the tail never counted, " + std::to_string(runs) +
                       " speculative runs equal" + first_bad);
    }
    // with the message: budgets 12 .. 30 reserve its 3 tokens and the lead's 1
    {
        bool ok = true;
        int32_t runs = 0;
        std::string first_bad;
        for (int32_t b = 12; b <= 30; ++b) {
            const config c = formed(with_msg(cfg(b, one, true)));
            const run_out o = scalar(of(own, c), c, 400);
            std::vector<int32_t> want = own.reason;
            int32_t counted = 20, forced = 0;
            if (b - 4 <= 20) {
                // at 20 free tokens the decision is the budget's: the message after the model's own 60, then the lead
                want = b - 4 <= 19 ? fillers(b - 4) : own.reason;
                want = then(want, {40, 41, 42, 60});
                counted = b;
                forced = 6;
            }
            want = then(want, {5, 61, 2000, 2001, tok_eos});
            const bool shape = o.g == want && o.counted == counted && o.added == counted && o.extra == 0 &&
                               o.forced == forced && !o.reasoning;
            int32_t r = 0;
            const bool sp = same_spec_m(of(own, c), c, 400, o, r, k1_more);
            runs += r;
            if ((!shape || !sp) && first_bad.empty()) {
                first_bad = " (first failure at budget " + std::to_string(b) + ": " + str(o.g) + ")";
            }
            ok = ok && shape && sp;
        }
        report(ok, "K1 with the message 40 41 42, budgets 12 .. 30: budget - 4 free tokens, then 40 41 42 60 5 61 "
                   "forced (6), counted = the budget; from 25 the model's own close, nothing forced, " +
                       std::to_string(runs) + " speculative runs equal" + first_bad);
    }
    // one forced token a step, the lead counted and the tail not; the tail is outside the reasoning
    {
        tracker tr;
        tr.reset(formed(cfg(3, one, true)));
        const int32_t fill[4] = {1100, 1101, 1102, 1103};
        bool ok = tr.observe(1000) == 1 && tr.room() == 1 && tr.observe(1001) == 1;
        ok = ok && tr.forced() == 60 && tr.cap(fill, 4) == 1 && tr.room() == 0 && tr.observe(60) == 1;
        ok = ok && tr.forced() == 5 && tr.cap(fill, 4) == 1 && tr.room() == 0 && tr.counted() == 3 &&
             tr.observe(5) == 0;
        ok = ok && !tr.reasoning() && tr.forced() == 61 && tr.cap(fill, 4) == 1 && tr.room() == 0 &&
             tr.observe(61) == 0;
        ok = ok && tr.forced() < 0 && tr.room() == INT32_MAX && tr.cap(fill, 4) == 5 && tr.counted() == 3 &&
             tr.forced_n() == 3 && tr.extra_n() == 0;
        report(ok, "K1 at budget 3: two free tokens, then 60 (counted: 3), 5 and, outside the reasoning, 61 (not "
                   "counted), each forced alone in its step; free after the tail");
    }
    // the model's own close under way: its closing prefix is completed alone and the tail follows; lead tokens the
    // reasoning already ends with are not fed twice
    {
        script f;
        f.false_at = 8;
        const config c2 = formed(cfg(10, two, true));
        const run_out o2 = scalar(f, c2, 400);
        int32_t runs = 0;
        const bool sp2 = same_spec_m(of(f, c2), c2, 400, o2, runs, k1_more);
        const bool own_close = o2.counted == 8 && o2.forced == 2 && o2.g[8] == 5 && o2.g[9] == 6 && o2.g[10] == 61 &&
                               std::find(o2.g.begin(), o2.g.end(), 60) == o2.g.end() && !o2.reasoning;
        // a two-token lead 60 62: the model's own 60 standing at the reserve (7 of budget 9) is completed with 62;
        // its own 60 62 there is the whole lead
        config c3 = cfg(9, one, true);
        c3.lead = {60, 62};
        c3.tail = {61};
        listed half, full;
        half.reason = then(fillers(6), {60, 1006, 1007, 1008, 1009});
        full.reason = then(fillers(5), {60, 62, 1007, 1008, 1009});
        half.answer = full.answer = {61, 2000};
        const run_out oh = scalar(of(half, c3), c3, 400), of2 = scalar(of(full, c3), c3, 400);
        const bool sp3 = same_spec_m(of(half, c3), c3, 400, oh, runs, k1_more) &&
                         same_spec_m(of(full, c3), c3, 400, of2, runs, k1_more);
        const bool halves = oh.g == then(fillers(6), {60, 62, 5, 61, 2000, tok_eos}) && oh.counted == 8 &&
                            oh.forced == 3 && of2.g == then(fillers(5), {60, 62, 5, 61, 2000, tok_eos}) &&
                            of2.counted == 7 && of2.forced == 2;
        // a message that ends with the lead's token: no second 60 (15 of 16 counted: the lead's reserve unused)
        config cm = formed(cfg(16, one, true));
        cm.message = {40, 41, 60};
        const run_out om = scalar(of(own, cm), cm, 400);
        const bool sp4 = same_spec_m(of(own, cm), cm, 400, om, runs, k1_more);
        const bool msg_lead = om.g == then(fillers(12), {40, 41, 60, 5, 61, 2000, 2001, tok_eos}) &&
                              om.counted == 15 && om.forced == 5 && om.extra == 0;
        report(own_close && sp2 && halves && sp3 && msg_lead && sp4,
               "K1 the model's own close under way: its 5 of a 5 6 close at the reserve completed with a forced 6, no "
               "60, then 61; its own 60 of a 60 62 lead completed with 62, its own 60 62 not fed again; a message "
               "ending with 60 is followed by the close directly, " +
                   std::to_string(runs) + " speculative runs equal: " + str(o2.g) + " | " + str(oh.g) + " | " +
                   str(om.g));
    }
    // reasoning reopened once the budget is spent: the whole form again, its lead fed and not counted; a total limit
    // inside the form ends the request there
    {
        script s2;
        s2.reopen_at = 2;
        const config c = formed(cfg(5, one, true));
        const run_out o = scalar(s2, c, 400);
        int32_t runs = 0;
        const bool sp = same_spec_m(of(s2, c), c, 400, o, runs, k1_more);
        // 4 free, 60 5 61, an answer token, the open, 60 5 61 at once
        const bool again = o.g.size() > 11 && o.g[4] == 60 && o.g[5] == 5 && o.g[6] == 61 && o.g[8] == tok_open &&
                           o.g[9] == 60 && o.g[10] == 5 && o.g[11] == 61 && o.counted == 5 && o.added == 5 &&
                           o.extra == 0 && o.forced == 6 && !o.reasoning;
        const config c1 = formed(cfg(3, one, true));
        const run_out l5 = scalar(of(own, c1), c1, 5), l4 = scalar(of(own, c1), c1, 4), l3 = scalar(of(own, c1), c1, 3);
        const bool sp2 = same_spec_m(of(own, c1), c1, 5, l5, runs, k1_more) &&
                         same_spec_m(of(own, c1), c1, 4, l4, runs, k1_more) &&
                         same_spec_m(of(own, c1), c1, 3, l3, runs, k1_more);
        const bool cut = l5.g == std::vector<int32_t>({1000, 1001, 60, 5, 61}) && !l5.reasoning &&
                         l4.g == std::vector<int32_t>({1000, 1001, 60, 5}) && !l4.reasoning && l4.forced == 2 &&
                         l3.g == std::vector<int32_t>({1000, 1001, 60}) && l3.reasoning && l3.counted == 3;
        report(again && sp && cut && sp2,
               "K1 reopened at a spent budget 5: 60 5 61 again at once, 5 counted and nothing extra; a total limit of "
               "5, 4 or 3 at budget 3 ends after the tail, after the close or inside the reasoning, " +
                   std::to_string(runs) + " speculative runs equal: " + str(o.g));
    }
    // the form's checks
    {
        std::string err;
        config c = formed(with_msg(cfg(13, two, true)));
        c.ends = {{8}};
        const bool taken = fn::think::check(c, err);
        c.lead = {60, 6};
        const bool lead_close = !fn::think::check(c, err);
        c.lead = {8};
        const bool lead_end = !fn::think::check(c, err);
        c.lead.assign(17, 60);
        const bool lead_long = !fn::think::check(c, err);
        c.lead = {60};
        c.tail = {61, tok_open};
        const bool tail_open = !fn::think::check(c, err);
        c.tail.assign(17, 61);
        const bool tail_long = !fn::think::check(c, err);
        report(taken && lead_close && lead_end && lead_long && tail_open && tail_long,
               "K1 check: a lead holding a token of the close or of another ending, a tail holding the open's, or "
               "either longer than 16 tokens refused; the test's taken");
    }
    // IFR1 R3: a budget spent inside a UTF-8 character, synthetic byte pieces (tokens 3000 ..; the filler 1000 .. and
    // answer 2000 .. are whole characters, not listed)
    {
        struct piece {
            int32_t tok;
            const char * bytes;
        };
        const piece pieces[] = {
            {3000, "\xC3"}, {3001, "\xA9"}, {3002, "\xE4"}, {3003, "\xB8\xAD"}, {3004, "\xF0"}, {3005, "\x9F"},
            {3006, "\x98"}, {3007, "\x80"}, {3010, ""}, {3011, "\xFF"}, {3013, "\xA9\xE4"}, {3014, "\xB8\xAD\xE6"},
            {3015, "a\xE4"}, {3020, "."}, {3021, "\n"},
        };
        const auto bytes_of = [&](int32_t t) {
            for (const piece & q : pieces) {
                if (q.tok == t) {
                    return std::string(q.bytes);
                }
            }
            return std::string(t == 5 || t == tok_open ? "" : "x"); // the sequences and the whole-character tokens
        };
        auto table = std::make_shared<std::vector<uint32_t>>();
        for (const piece & q : pieces) {
            table->push_back(fn::think::utf8_entry(q.tok, q.bytes, std::string(q.bytes).size()));
        }
        // the classes (bits: leading continuation bytes, the last character's missing bytes << 3, 32 nothing after
        // the leading ones, 64 malformed)
        const auto k = [](const char * b) {
            return static_cast<int32_t>(fn::think::utf8_class(b, std::string(b).size()));
        };
        const bool classes = k("a") == 0 && k("\xC3\xA9") == 0 && k("\xF0\x9F\x98\x80") == 0 && k("\xC3") == 8 &&
                             k("\xF0") == 24 && k("\xA9") == 33 && k("\xB8\xAD") == 34 && k("\xAD\xE6") == 17 &&
                             k("") == 32 && k("\xFF") == 64 && k("a\xE4") == 16 && k("\xE4" "a") == 64 &&
                             fn::think::utf8_entry(7, "a", 1) == 0 &&
                             fn::think::utf8_entry(3000, "\xC3", 1) == (3000u | 8u << 24);
        report(classes, "utf8 classes: whole characters 0 (unlisted); a lead byte, continuation bytes, a token "
                        "finishing one character and beginning another, no bytes, malformed bytes");
        {
            std::string err;
            config c = cfg(4, one, true);
            using table_t = std::vector<uint32_t>;
            c.utf8 = std::make_shared<table_t>(table_t{3001u | 33u << 24, 3000u | 8u << 24});
            const bool unsorted = !fn::think::check(c, err);
            c.utf8 = std::make_shared<table_t>(table_t{3000u});
            const bool whole = !fn::think::check(c, err);
            c.utf8 = table;
            report(unsorted && whole && fn::think::check(c, err),
                   "check: a UTF-8 table out of order or listing a whole-character token refused, the test's taken");
        }
        const std::vector<int32_t> more = {3000, 3001, 3002, 3003, 3004, 3005, 3010, 3011, 3013, 3014};
        // the reasoning's bytes up to the close (or to the end): a whole sequence of characters?
        const auto whole_before_close = [&](const run_out & o) {
            std::string b;
            for (const int32_t t : o.g) {
                if (t == 5) {
                    break;
                }
                b += bytes_of(t);
            }
            return fn::think::utf8_class(b.data(), b.size()) == 0;
        };
        struct ucase {
            const char * name;
            std::vector<int32_t> reason;
            int32_t budget, limit, extra, forced;
            bool whole, closed;
        };
        const std::vector<int32_t> f4 = {1000, 1001, 1002, 1003};
        const auto r = [&](std::vector<int32_t> tail) {
            std::vector<int32_t> v = f4;
            v.insert(v.end(), tail.begin(), tail.end());
            return v;
        };
        const ucase ucases[] = {
            {"2-byte, the budget after its lead byte: 1 extra token", r({3000, 3001, 1004, 1005}), 5, 400, 1, 1, true,
             true},
            {"3-byte as 1 + 2 bytes, the budget after the first: 1 extra", r({3002, 3003, 1004}), 5, 400, 1, 1, true,
             true},
            {"4-byte as four bytes, the budget after the first: 3 extra (the most)",
             r({3004, 3005, 3006, 3007, 1004}), 5, 400, 3, 1, true, true},
            {"4-byte, the budget after its second byte: 2 extra", r({3004, 3005, 3006, 3007, 1004}), 6, 400, 2, 1, true,
             true},
            {"a token without bytes while waiting: 2 extra", r({3002, 3010, 3003, 1004}), 5, 400, 2, 1, true, true},
            {"a token finishing one character and beginning one at the budget: 1 extra", {1000, 1001, 1002, 3015, 3003,
             1004}, 4, 400, 1, 1, true, true},
            {"the budget on a character boundary: nothing extra", r({3000, 3001, 1004}), 6, 400, 0, 1, true, true},
            {"ASCII breaks the character: the wait ends after it (1 extra, malformed bytes kept)",
             r({3004, 1100, 1004}), 5, 400, 1, 1, false, true},
            {"a malformed byte while waiting: the wait ends after it", r({3002, 3011, 1004}), 5, 400, 1, 1, false,
             true},
            {"the cap: three tokens each finishing a character and beginning another, the close forced after them",
             r({3000, 3013, 3014, 3014, 1004}), 5, 400, 3, 1, false, true},
            {"the model's own close while waiting: nothing forced or extra", r({3000}), 5, 400, 0, 0, false, true},
            {"the total limit while waiting: the reasoning unclosed, 2 extra", r({3004, 3005, 3006, 3007}), 5, 7, 2, 0,
             false, false},
        };
        for (const ucase & u : ucases) {
            listed l;
            l.reason = u.reason;
            l.answer = {2000, 2001};
            config c = cfg(u.budget, one, true);
            c.utf8 = table;
            const run_out o = scalar(of(l, c), c, u.limit);
            config c0 = c;
            c0.utf8 = nullptr;
            const run_out b = scalar(of(l, c0), c0, u.limit);
            const auto at = static_cast<size_t>(u.budget + u.extra);
            const bool closed = !o.reasoning && o.g.size() > at && o.g[at] == 5;
            bool ok = o.counted == u.budget && o.extra == u.extra && o.forced == u.forced && closed == u.closed &&
                      whole_before_close(o) == u.whole && o.added == u.budget;
            // the old tracker (no table) closes at the budget itself
            ok = ok && b.extra == 0 && (u.forced == 0 || (b.g.size() > static_cast<size_t>(u.budget) &&
                                                         b.g[static_cast<size_t>(u.budget)] == 5));
            int32_t runs = 0;
            ok = ok && same_spec_m(of(l, c), c, u.limit, o, runs, more);
            report(ok, std::string("utf8: ") + u.name + ", " + std::to_string(runs) + " speculative runs equal: " +
                           str(o.g) + " | no table: " + str(b.g));
        }
        // ASCII reasoning: the table changes nothing
        {
            script s;
            config c = cfg(10, one, true), c0 = c;
            c.utf8 = table;
            const run_out o = scalar(s, c, 400), b = scalar(s, c0, 400);
            report(o.g == b.g && o.extra == 0 && o.counted == 10,
                   "utf8: ASCII reasoning, budget 10: the same tokens with and without the table");
        }
        // FBH1 A1: the wait before the message: the reserve (budget 20 less 3) lands after a 4-byte character's first
        // byte: its 3 bytes extra, then the message and the close; 20 counted, 23 reasoning tokens in all
        {
            listed l;
            for (int32_t i = 0; i < 16; ++i) {
                l.reason.push_back(1000 + i);
            }
            l.reason.insert(l.reason.end(), {3004, 3005, 3006, 3007, 1016, 1017});
            l.answer = {2000, 2001};
            config c = with_msg(cfg(20, one, true));
            c.utf8 = table;
            const run_out o = scalar(of(l, c), c, 400);
            int32_t runs = 0;
            std::vector<int32_t> alphabet = more;
            alphabet.insert(alphabet.end(), msg.begin(), msg.end());
            const bool sp = same_spec_m(of(l, c), c, 400, o, runs, alphabet);
            const bool shape = o.g.size() == 20 + 3 + 1 + 2 + 1 && o.g[16] == 3004 && o.g[19] == 3007 &&
                               o.g[20] == 40 && o.g[22] == 42 && o.g[23] == 5 && o.counted == 20 && o.extra == 3 &&
                               o.forced == 4 && o.added == 20 && !o.reasoning && whole_before_close(o);
            // the character complete at the reserve (its 4 bytes are tokens 14 .. 17): nothing extra
            listed l2 = l;
            l2.reason.erase(l2.reason.begin(), l2.reason.begin() + 3);
            const run_out o2 = scalar(of(l2, c), c, 400);
            const bool whole = o2.extra == 0 && o2.counted == 20 && o2.g[17] == 40 && o2.g[20] == 5;
            report(shape && sp && whole,
                   "utf8: the message waits for a character's bytes at the reserve (17 of budget 20): 3 extra, then 40 "
                   "41 42 and the close, 20 counted; nothing extra on a boundary, " + std::to_string(runs) +
                       " speculative runs equal: " + str(o.g));
        }
        // FBH1 L1: the sentence-end wait. Budget 40 and the message 40 41 42: the reserve is at 37 and the window is
        // the 10 free tokens before it (counts 27 .. 36); 3020 is "." and 3021 a newline
        {
            const auto se = [](const char * b) { return fn::think::sentence_end(b, std::string(b).size()); };
            const bool texts = se(".") && se("x? ") && se("!") && se("\n") && se("x.\n\n") && se("\xE3\x80\x82") &&
                               se("a\xEF\xBC\x9F ") && !se("x") && !se(". x") && !se("") && !se(" ") && !se(":") &&
                               fn::think::utf8_entry(3020, ".", 1) == (3020u | 128u << 24) &&
                               fn::think::utf8_entry(3021, "\n", 1) == (3021u | 128u << 24) &&
                               fn::think::utf8_entry(3022, ".\xC3", 2) == (3022u | 8u << 24) &&
                               fn::think::wait_window(40) == 10 && fn::think::wait_window(1000) == 64;
            struct scase {
                const char * name;
                std::vector<int32_t> ends; // the reasoning indexes holding a sentence end (3020, or 3021 if negated)
                bool message, tabled;
                int32_t budget;
                int32_t at;      // where the forced tokens begin
                int32_t counted; // reasoning tokens counted
            };
            const scase scases[] = {
                {"a sentence end before the window only: the message at the reserve", {10}, true, true, 40, 37, 40},
                {"a sentence end inside the window: the message right after it", {10, 30}, true, true, 40, 31, 34},
                {"two inside the window: the first one", {28, 33}, true, true, 40, 29, 32},
                {"the window's first token (count 27)", {26}, true, true, 40, 27, 30},
                {"the token before the window (count 26): no early message", {25}, true, true, 40, 37, 40},
                {"a newline as the last free token: the reserve as before", {-36}, true, true, 40, 37, 40},
                {"a newline inside the window", {-32}, true, true, 40, 33, 36},
                {"no message: the close at the budget whatever ends a sentence", {30, 35}, false, true, 40, 40, 40},
                {"no table: the message at the reserve", {30}, true, false, 40, 37, 40},
                {"budget 12 (window 3, counts 6 .. 8): a sentence end at count 7", {6}, true, true, 12, 7, 10},
            };
            bool ok = texts;
            int32_t runs = 0;
            std::string first_bad;
            std::vector<int32_t> alphabet = more;
            alphabet.insert(alphabet.end(), msg.begin(), msg.end());
            alphabet.insert(alphabet.end(), {3020, 3021});
            for (const scase & q : scases) {
                listed l;
                for (int32_t i = 0; i < 60; ++i) {
                    l.reason.push_back(1000 + i);
                }
                for (const int32_t e : q.ends) {
                    l.reason[static_cast<size_t>(e < 0 ? -e : e)] = e < 0 ? 3021 : 3020;
                }
                l.answer = {2000, 2001};
                config c = cfg(q.budget, one, true);
                if (q.message) {
                    c = with_msg(c);
                }
                if (q.tabled) {
                    c.utf8 = table;
                }
                const run_out o = scalar(of(l, c), c, 400);
                const auto at = static_cast<size_t>(q.at);
                const size_t n_msg = q.message ? 3 : 0;
                bool shape = !o.reasoning && o.extra == 0 && o.counted == q.counted && o.added == q.counted &&
                             o.forced == static_cast<int32_t>(n_msg) + 1 && o.g.size() == at + n_msg + 1 + 2 + 1 &&
                             o.g[at + n_msg] == 5 && o.counted <= q.budget;
                shape = shape && (!q.message || (o.g[at] == 40 && o.g[at + 2] == 42));
                for (size_t i = 0; shape && i < at; ++i) {
                    shape = o.g[i] == l.reason[i];
                }
                int32_t r = 0;
                const bool sp = same_spec_m(of(l, c), c, 400, o, r, alphabet);
                runs += r;
                if ((!shape || !sp) && first_bad.empty()) {
                    first_bad = std::string(" (first failure: ") + q.name + ": " + str(o.g) + ")";
                }
                ok = ok && shape && sp;
            }
            // the engine's step sizes: one free position at a time inside the window, none once the message is due
            config c = with_msg(cfg(40, one, true));
            c.utf8 = table;
            tracker tr;
            tr.reset(c);
            for (int32_t i = 0; i < 20; ++i) {
                tr.observe(1000 + i);
            }
            const int32_t props[4] = {1020, 1021, 1022, 1023};
            const bool before = tr.room() == 7 && tr.cap(props, 4) == 5;
            for (int32_t i = 20; i < 28; ++i) {
                tr.observe(1000 + i);
            }
            const int32_t ends[4] = {1028, 3020, 1030, 1031};
            const bool inside = tr.forced() < 0 && tr.room() == 1 && tr.cap(props, 4) == 5 && tr.cap(ends, 4) == 2;
            tr.observe(1028);
            tr.observe(3020);
            const bool due = tr.forced() == 40 && tr.room() == 0 && tr.cap(props, 4) == 1 && tr.counted() == 30;
            report(ok && before && inside && due,
                   "sentence end: the texts that end one; budget 40 and 12 with the message: the message right after "
                   "the first sentence end in the window before the reserve, else at the reserve; no message or no "
                   "table as before; room 1 inside the window, " + std::to_string(runs) + " speculative runs equal" +
                       first_bad);
        }
        // FBH1 K1 with the table: the wait for a character's bytes comes before the lead, and a sentence's end starts
        // the message early with the lead after it; the newline 3021 is the lead here, as in the model
        {
            int32_t runs = 0;
            std::vector<int32_t> alphabet = more;
            alphabet.insert(alphabet.end(), {40, 41, 42, 60, 61, 3020, 3021});
            // budget 6: the reserve (5) lands after a 4-byte character's first byte: its 3 bytes extra, then 60 5 61
            listed lw;
            lw.reason = r({3004, 3005, 3006, 3007, 1004});
            lw.answer = {61, 2000};
            config cw = formed(cfg(6, one, true));
            cw.utf8 = table;
            const run_out ow = scalar(of(lw, cw), cw, 400);
            const bool sp1 = same_spec_m(of(lw, cw), cw, 400, ow, runs, alphabet);
            const bool waits = ow.g == std::vector<int32_t>({1000, 1001, 1002, 1003, 3004, 3005, 3006, 3007, 60, 5, 61,
                                                             2000, tok_eos}) &&
                               ow.counted == 6 && ow.extra == 3 && ow.added == 6 && ow.forced == 3 &&
                               whole_before_close(ow);
            // budget 40 with the message: the reserve is at 36 and the window the 10 free tokens before it; a sentence
            // end at count 30 starts the message, then the lead 3021, the close and the tail
            listed ls;
            for (int32_t i = 0; i < 60; ++i) {
                ls.reason.push_back(1000 + i);
            }
            ls.reason[29] = 3020;
            ls.answer = {61, 2000};
            config cs = with_msg(cfg(40, one, true));
            cs.lead = {3021};
            cs.tail = {61};
            cs.utf8 = table;
            const run_out os = scalar(of(ls, cs), cs, 400);
            const bool sp2 = same_spec_m(of(ls, cs), cs, 400, os, runs, alphabet);
            const bool early = os.g.size() == 30 + 3 + 3 + 1 + 1 && os.g[29] == 3020 && os.g[30] == 40 &&
                               os.g[32] == 42 && os.g[33] == 3021 && os.g[34] == 5 && os.g[35] == 61 &&
                               os.counted == 34 && os.extra == 0 && os.forced == 6;
            // no sentence end: the message at the reserve, counted = the budget
            listed ln = ls;
            ln.reason[29] = 1029;
            const run_out on = scalar(of(ln, cs), cs, 400);
            const bool sp3 = same_spec_m(of(ln, cs), cs, 400, on, runs, alphabet);
            const bool late = on.g.size() == 36 + 3 + 3 + 1 + 1 && on.g[36] == 40 && on.g[39] == 3021 &&
                              on.g[40] == 5 && on.g[41] == 61 && on.counted == 40 && on.forced == 6;
            // without a message (budget 40, the reserve at 39): the model's own newline as the last free token is
            // the lead, so the close follows it directly
            listed lo = ln;
            lo.reason[38] = 3021;
            config co = cfg(40, one, true);
            co.lead = {3021};
            co.tail = {61};
            co.utf8 = table;
            const run_out oo = scalar(of(lo, co), co, 400);
            const bool sp4 = same_spec_m(of(lo, co), co, 400, oo, runs, alphabet);
            const bool own_nl = oo.g.size() == 39 + 2 + 1 + 1 && oo.g[38] == 3021 && oo.g[39] == 5 && oo.g[40] == 61 &&
                                oo.counted == 39 && oo.forced == 2;
            report(waits && sp1 && early && sp2 && late && sp3 && own_nl && sp4,
                   "K1 with the table: a character's 3 bytes waited for before the lead (6 counted + 3 extra, the "
                   "budget + 3); the message right after a sentence's end, then the lead (34 of 40 counted), else at "
                   "the reserve (40); the model's own newline as the last free token is not fed twice, " +
                       std::to_string(runs) + " speculative runs equal: " + str(ow.g));
        }
        // a slot reused in the middle of a wait starts fresh; two slots sharing the table stepped in turn
        {
            listed la, lb;
            la.reason = r({3004, 3005, 3006, 3007, 1004});
            lb.reason = r({3000, 3013, 3014, 3014, 1004});
            la.answer = lb.answer = {2000, 2001};
            config ca = cfg(5, one, true), cb = cfg(5, one, true);
            ca.utf8 = cb.utf8 = table;
            tracker tr;
            tr.reset(ca);
            for (int32_t i = 0; i < 6; ++i) {
                tr.observe(la.reason[static_cast<size_t>(i)]);
            }
            const bool mid = tr.forced() < 0 && tr.extra_n() == 1;
            tr.reset(cb);
            run_out o;
            while (!done(o, 400)) {
                const int32_t f = tr.forced();
                emit(tr, o, f >= 0 ? f : of(lb, cb)(o.g), false);
            }
            o = finish(tr, o);
            const run_out wb = scalar(of(lb, cb), cb, 400), wa = scalar(of(la, ca), ca, 400);
            tracker ta, tb2;
            ta.reset(ca);
            tb2.reset(cb);
            run_out oa, ob;
            std::mt19937 ra(21), rb(22);
            while (!done(oa, 400) || !done(ob, 400)) {
                if (!done(oa, 400)) {
                    spec_step(of(la, ca), ca, ta, oa, 400, 4, ra, more);
                }
                if (!done(ob, 400)) {
                    spec_step(of(lb, cb), cb, tb2, ob, 400, 3, rb, more);
                }
            }
            oa = finish(ta, oa);
            ob = finish(tb2, ob);
            report(mid && o.g == wb.g && o.extra == wb.extra && oa.g == wa.g && ob.g == wb.g && oa.extra == 3 &&
                       ob.extra == 3 && oa.bad == 0 && ob.bad == 0,
                   "utf8: a slot reused in the middle of a wait runs fresh; two slots sharing the table, stepped in "
                   "turn, each equal to its lone run");
        }
        // reasoning reopened after a wait the model's own close ended: a spent budget forces at once
        {
            config c = cfg(1, one, true);
            c.utf8 = table;
            tracker tr;
            tr.reset(c);
            tr.observe(3000);
            const bool waits = tr.forced() < 0 && tr.room() == 1;
            tr.observe(5);
            tr.observe(2000);
            tr.observe(tok_open);
            report(waits && tr.reasoning() && tr.forced() == 5 && tr.counted() == 1,
                   "utf8: reopened after a wait the model's close ended: the close forced at once");
        }
    }
    // IFR1 R3: the engines' fixtures (the model's tokenizer and the server's table)
    for (const std::string & path : fixtures) {
        fixture f;
        std::string err;
        if (!read_fixture(path, f, err)) {
            report(false, "fixture " + path + ": " + err);
            continue;
        }
        std::vector<int32_t> more = f.c.close;
        for (size_t i = 0; i < f.c.utf8->size() && i < 12; ++i) {
            more.push_back(static_cast<int32_t>((*f.c.utf8)[i] & ((1u << 24) - 1)));
        }
        for (const fixture_case & fc : f.cases) {
            config c = f.c;
            c.budget = fc.budget;
            config c0 = c;
            c0.utf8 = nullptr;
            const run_out o = scalar(of(fc.l, c), c, fc.limit, fc.l.eos);
            const run_out b = scalar(of(fc.l, c0), c0, fc.limit, fc.l.eos);
            const auto same = [](const run_out & x, const run_out & y) {
                return x.g == y.g && x.counted == y.counted && x.forced == y.forced && x.extra == y.extra &&
                       x.reasoning == y.reasoning;
            };
            int32_t runs = 0;
            const bool sp = same_spec_m(of(fc.l, c), c, fc.limit, o, runs, more);
            report(same(o, fc.want) && same(b, fc.base) && sp && o.bad == 0,
                   "fixture " + f.model + " " + fc.name + ": budget " + std::to_string(fc.budget) + ", counted " +
                       std::to_string(o.counted) + ", extra " + std::to_string(o.extra) + ", forced " +
                       std::to_string(o.forced) + (o.reasoning ? ", unclosed" : "") + ", " + std::to_string(runs) +
                       " speculative runs equal: " + str(o.g) + " | no table: " + str(b.g));
        }
    }
    std::printf("think: %s (%d failed)\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails;
}

} // namespace fn::tools
