#include "fn/runtime/think_budget.h"

#include <algorithm>

namespace fn::think {

namespace {

constexpr size_t max_seq = 16;

// the longest prefix of seq that ends the matched prefix seq[0 .. m - 1] followed by t (the KMP step; the sequences
// are a few tokens, so a direct search): the opening sequence's and the forced close's lead's
int32_t match_next(const std::vector<int32_t> & seq, int32_t m, int32_t t) {
    for (int32_t k = std::min<int32_t>(m + 1, static_cast<int32_t>(seq.size())); k > 0; --k) {
        // seq[0 .. k - 1] == (seq[m - k + 1 .. m - 1], t)
        bool ok = seq[static_cast<size_t>(k - 1)] == t;
        for (int32_t j = 0; ok && j + 1 < k; ++j) {
            ok = seq[static_cast<size_t>(j)] == seq[static_cast<size_t>(m - k + 1 + j)];
        }
        if (ok) {
            return k;
        }
    }
    return 0;
}

// the longest k <= max_k with s's last k tokens seq[0 .. k - 1] (s has n tokens)
int32_t suffix_prefix(const std::vector<int32_t> & s, const std::vector<int32_t> & seq, int32_t max_k) {
    const auto n = static_cast<int32_t>(s.size());
    for (int32_t k = std::min({max_k, n, static_cast<int32_t>(seq.size())}); k > 0; --k) {
        if (std::equal(seq.begin(), seq.begin() + k, s.end() - k)) {
            return k;
        }
    }
    return 0;
}

constexpr uint32_t token_mask = (1u << 24) - 1;

// the continuation bytes the text's last character lacks after a token of class k (it lacked need before); false: the
// token's bytes are not UTF-8 there (need restarts at the token's last character)
bool utf8_next(int32_t & need, uint8_t k) {
    const int32_t lead = k & 7;
    if ((k & 32) != 0) {
        if (lead <= need) {
            need -= lead;
            return true;
        }
        need = 0;
        return false;
    }
    const bool ok = lead == need && (k & 64) == 0;
    need = (k >> 3) & 3;
    return ok;
}

} // namespace

uint8_t utf8_class(const char * bytes, size_t n) {
    // the bytes as unsigned values, the same object representation
    const auto * b = reinterpret_cast<const unsigned char *>(bytes);
    size_t i = 0;
    int32_t lead = 0;
    while (i < n && (b[i] & 0xC0) == 0x80) {
        ++i;
        ++lead;
    }
    const auto lead_bits = static_cast<uint32_t>(std::min(lead, 7));
    if (i == n) {
        return static_cast<uint8_t>(lead_bits | 32u);
    }
    int32_t need = 0;
    bool bad = false;
    for (; i < n; ++i) {
        const uint32_t c = b[i];
        if ((c & 0xC0) == 0x80) {
            bad = bad || need == 0;
            need = std::max(0, need - 1);
            continue;
        }
        bad = bad || need > 0;
        // C0, C1 and F5 .. FF never begin a character
        need = c < 0x80 ? 0 : c >= 0xC2 && c <= 0xDF ? 1 : c >= 0xE0 && c <= 0xEF ? 2 : c >= 0xF0 && c <= 0xF4 ? 3 : -1;
        bad = bad || need < 0;
        need = std::max(need, 0);
    }
    return static_cast<uint8_t>(lead_bits | static_cast<uint32_t>(need) << 3 | (bad ? 64u : 0u));
}

bool sentence_end(const char * bytes, size_t n) {
    if (n > 0 && bytes[n - 1] == '\n') {
        return true;
    }
    while (n > 0 && bytes[n - 1] == ' ') {
        --n;
    }
    if (n == 0) {
        return false;
    }
    const char c = bytes[n - 1];
    if (c == '.' || c == '!' || c == '?') {
        return true;
    }
    // U+3002, U+FF01, U+FF1F
    const std::string last(bytes + (n >= 3 ? n - 3 : 0), bytes + n);
    return last == "\xE3\x80\x82" || last == "\xEF\xBC\x81" || last == "\xEF\xBC\x9F";
}

uint32_t utf8_entry(int32_t token, const char * bytes, size_t n) {
    uint8_t k = utf8_class(bytes, n);
    // a sentence ends on a whole character: a token whose bytes are malformed or end inside one ends none
    if ((k & (64u | 24u)) == 0 && sentence_end(bytes, n)) {
        k |= class_sentence;
    }
    if (k == 0 || token < 0 || static_cast<uint32_t>(token) > token_mask) {
        return 0;
    }
    return static_cast<uint32_t>(token) | static_cast<uint32_t>(k) << 24;
}

bool check(const config & c, std::string & err) {
    if (c.budget < -1) {
        err = "the reasoning budget " + std::to_string(c.budget) + " is below -1";
        return false;
    }
    if (c.budget >= 0 && c.close.empty()) {
        err = "a reasoning budget needs the model's closing sequence (the chat template has none)";
        return false;
    }
    if (c.open.size() > max_seq || c.close.size() > max_seq || c.lead.size() > max_seq || c.tail.size() > max_seq) {
        err = "the reasoning sequences are longer than 16 tokens";
        return false;
    }
    for (const std::vector<int32_t> & e : c.ends) {
        if (e.empty() || e.size() > max_seq) {
            err = "a reasoning ending is empty or longer than 16 tokens";
            return false;
        }
    }
    if (c.utf8) {
        for (size_t i = 0; i < c.utf8->size(); ++i) {
            const uint32_t e = (*c.utf8)[i];
            if ((e >> 24) == 0 || (i > 0 && (e & token_mask) <= ((*c.utf8)[i - 1] & token_mask))) {
                err = "the reasoning's UTF-8 table is not ascending by token";
                return false;
            }
        }
    }
    if (c.message.size() > static_cast<size_t>(max_message)) {
        err = "the reasoning budget message is longer than " + std::to_string(max_message) + " tokens";
        return false;
    }
    if (!c.message.empty() && !c.close.empty()) {
        // the message then the close, as fed: no ending may complete before the close's last token
        std::vector<int32_t> fed = c.message;
        fed.insert(fed.end(), c.close.begin(), c.close.end());
        std::vector<std::vector<int32_t>> all = c.ends;
        all.push_back(c.close);
        for (const std::vector<int32_t> & e : all) {
            const auto last = std::find_end(fed.begin(), fed.end() - 1, e.begin(), e.end());
            if (last != fed.end() - 1) {
                err = "the reasoning budget message holds a sequence that ends reasoning";
                return false;
            }
        }
        if (suffix_prefix(c.message, c.close, static_cast<int32_t>(c.close.size()) - 1) > 0) {
            err = "the reasoning budget message ends with tokens that begin the closing sequence";
            return false;
        }
    }
    // K1: a lead token never extends an ending (so it is counted, or dropped, the moment it is fed), and the tail
    // never opens reasoning
    const auto holds = [](const std::vector<int32_t> & seq, int32_t t) {
        return std::find(seq.begin(), seq.end(), t) != seq.end();
    };
    for (const int32_t t : c.lead) {
        bool in_ending = holds(c.close, t);
        for (const std::vector<int32_t> & e : c.ends) {
            in_ending = in_ending || holds(e, t);
        }
        if (in_ending) {
            err = "the forced close's lead holds a token of a sequence that ends reasoning";
            return false;
        }
    }
    for (const int32_t t : c.tail) {
        if (holds(c.open, t)) {
            err = "the forced close's tail holds a token of the opening sequence";
            return false;
        }
    }
    return true;
}

void tracker::reset(const config & c) {
    c_ = c;
    in_ = tracks() && c.open_at_start;
    msg_on_ = c.budget >= 0 && reserve_ok(c.budget, c.message.size());
    msg_ = 0;
    lead_m_ = 0;
    closing_ = false;
    tail_ = -1;
    sent_ = false;
    body_ = 0;
    pend_.clear();
    om_ = 0;
    n_forced_ = 0;
    need_ = 0;
    waited_ = 0;
    broken_ = false;
    n_extra_ = 0;
}

uint8_t tracker::class_of(int32_t token) const {
    const std::vector<uint32_t> & t = *c_.utf8;
    const auto it = std::lower_bound(t.begin(), t.end(), static_cast<uint32_t>(token),
                                     [](uint32_t e, uint32_t k) { return (e & token_mask) < k; });
    return it != t.end() && (*it & token_mask) == static_cast<uint32_t>(token) ? static_cast<uint8_t>(*it >> 24) : 0;
}

bool tracker::waiting() const {
    return c_.utf8 && need_ > 0 && waited_ < max_extra && !broken_ && msg_ == 0 && !closing_;
}

int32_t tracker::limit() const {
    return c_.budget - static_cast<int32_t>(c_.lead.size()) -
           (msg_on_ && msg_ == 0 ? static_cast<int32_t>(c_.message.size()) : 0);
}

int32_t tracker::window() const {
    return msg_on_ && c_.utf8 ? wait_window(c_.budget) : 0;
}

bool tracker::early() const {
    // not while the model's own tokens may be the closing sequence under way
    return msg_on_ && msg_ == 0 && sent_ && need_ == 0 && pend_.empty() && body_ >= limit() - window();
}

int32_t tracker::forced() const {
    part p = part::none;
    return forced_from(p);
}

int32_t tracker::forced_from(part & p) const {
    p = part::none;
    if (!active()) {
        return -1;
    }
    if (!in_) {
        // K1: a forced close's tail, outside the reasoning
        if (tail_ < 0) {
            return -1;
        }
        p = part::tail;
        return c_.tail[static_cast<size_t>(tail_)];
    }
    if (msg_ > 0 && msg_ < static_cast<int32_t>(c_.message.size())) {
        p = part::message; // the message under way
        return c_.message[static_cast<size_t>(msg_)];
    }
    // a fed message or a lead under way is followed by the close whatever the count
    // FBH1 L1: free below the limit, unless the message starts early at a sentence's end
    if (msg_ == 0 && !closing_ && !early() && (body_ + static_cast<int32_t>(pend_.size()) < limit() || waiting())) {
        return -1;
    }
    // the close continues from the closing prefix the waiting tokens end with (they may begin another ending too)
    const int32_t j = suffix_prefix(pend_, c_.close, static_cast<int32_t>(c_.close.size()) - 1);
    if (j > 0) {
        p = part::close;
        return c_.close[static_cast<size_t>(j)];
    }
    if (msg_on_ && msg_ == 0) {
        p = part::message;
        return c_.message[0];
    }
    // K1: the lead first, from the part of it the reasoning already ends with
    if (lead_m_ < static_cast<int32_t>(c_.lead.size())) {
        p = part::lead;
        return c_.lead[static_cast<size_t>(lead_m_)];
    }
    p = part::close;
    return c_.close[0];
}

int32_t tracker::cap(const int32_t * props, int32_t n) const {
    if (!active()) {
        return n + 1;
    }
    tracker x = *this;
    for (int32_t p = 0; p <= n; ++p) {
        if (x.forced() >= 0) {
            return std::max(p, 1);
        }
        if (p < n) {
            x.observe(props[p]);
        }
    }
    return n + 1;
}

int32_t tracker::room() const {
    if (!active()) {
        return INT32_MAX;
    }
    if (!in_) {
        return tail_ >= 0 ? 0 : INT32_MAX;
    }
    if (msg_ > 0 || closing_) {
        return 0; // the message or the lead under way or fed: every position is forced
    }
    if (early()) {
        return 0;
    }
    const int32_t left = limit() - body_ - static_cast<int32_t>(pend_.size());
    if (left <= 0) {
        // a spent budget waiting for a character's bytes: this position is free, the next may be forced
        return waiting() ? 1 : 0;
    }
    // FBH1 L1: inside the sentence-end window any token may end a sentence, so only this position is surely free
    return std::max(1, left - window());
}

int32_t tracker::observe(int32_t t) {
    if (!tracks()) {
        return 0;
    }
    const auto lo = static_cast<int32_t>(c_.open.size());
    if (in_) {
        part p = part::none;
        const int32_t f = forced_from(p);
        const bool msg = f == t && p == part::message;
        const bool lead = f == t && p == part::lead;
        if (f == t) {
            ++n_forced_;
        }
        if (c_.utf8 && active()) {
            const uint8_t k = class_of(t);
            const bool ok = utf8_next(need_, k);
            if (f < 0 && body_ + static_cast<int32_t>(pend_.size()) >= limit()) {
                ++waited_;
                broken_ = broken_ || !ok;
            }
            sent_ = f < 0 && (k & class_sentence) != 0;
        }
        // the free reasoning's cap before this token (the message's tokens count up to the budget itself)
        const int32_t cap = msg ? c_.budget : limit();
        if (msg) {
            ++msg_;
        }
        closing_ = closing_ || lead;
        lead_m_ = match_next(c_.lead, lead_m_, t);
        // tokens leaving the wait: counted up to the cap, past it extra (finishing a character); no budget: counted.
        // K1: a forced lead token is the last of them (check: it extends no ending); it is counted when the budget has
        // room for it and otherwise not at all
        const auto take = [&](int32_t add) {
            const int32_t rest = add - (lead ? 1 : 0);
            int32_t b = active() ? std::min(rest, std::max(0, cap - body_)) : rest;
            n_extra_ += rest - b;
            b += lead && body_ + b < c_.budget ? 1 : 0;
            body_ += b;
            return b;
        };
        pend_.push_back(t);
        const auto n = static_cast<int32_t>(pend_.size());
        // an ending completed (the longest one, if several): the tokens before it are reasoning
        int32_t done = 0;
        int32_t keep = 0; // else the longest waiting suffix that still begins an ending
        const auto try_seq = [&](const std::vector<int32_t> & e) {
            const auto le = static_cast<int32_t>(e.size());
            if (le <= n && std::equal(e.begin(), e.end(), pend_.end() - le)) {
                done = std::max(done, le);
            }
            keep = std::max(keep, suffix_prefix(pend_, e, le - 1));
        };
        try_seq(c_.close);
        for (const std::vector<int32_t> & e : c_.ends) {
            try_seq(e);
        }
        if (done > 0) {
            const int32_t add = take(n - done);
            pend_.clear();
            in_ = false;
            om_ = 0;
            need_ = 0;
            waited_ = 0;
            broken_ = false;
            sent_ = false;
            lead_m_ = 0;
            closing_ = false;
            // K1: the tail after a close whose last token the budget forced
            tail_ = f == t && p == part::close && !c_.tail.empty() ? 0 : -1;
            return add;
        }
        const int32_t add = n - keep;
        pend_.erase(pend_.begin(), pend_.begin() + add);
        return take(add);
    }
    if (tail_ >= 0) {
        // K1: a tail token (forced, never counted); any other token ends the tail
        const bool hit = active() && c_.tail[static_cast<size_t>(tail_)] == t;
        n_forced_ += hit ? 1 : 0;
        tail_ = hit && tail_ + 1 < static_cast<int32_t>(c_.tail.size()) ? tail_ + 1 : -1;
    }
    if (lo == 0) {
        return 0;
    }
    om_ = match_next(c_.open, om_, t);
    if (om_ == lo) {
        in_ = true;
        om_ = 0;
        pend_.clear();
        need_ = 0;
        waited_ = 0;
        broken_ = false;
        sent_ = false;
        lead_m_ = 0;
        closing_ = false;
        tail_ = -1;
    }
    return 0;
}

} // namespace fn::think
