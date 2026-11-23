#pragma once

// A request's reasoning budget (GLFL16 thinking-budget order): at most `budget` generated reasoning tokens in this
// assistant turn, then the model's real closing sequence, fed through the model like any generated token, and the
// answer after it. CPU only; each engine's step asks it, before a pass, how many tokens the pass may emit and whether
// its first token is forced, and tells it every token it emits (accepted proposals and the token after them; never a
// rejected proposal, the prompt or another slot's tokens).
//
// Accounting. A token emitted while reasoning is open counts against the budget, except the tokens of the closing
// sequence (natural or forced), of any other sequence the model's parser ends reasoning with (config::ends: Qwen's
// <tool_call>), and of the opening sequence. Tokens that may still begin one of those endings wait; they are counted
// once no ending can match them any more (the model went on reasoning). So at most `budget` tokens are counted, and
// when the count reaches the budget every further position in the reasoning is the closing sequence's next token: the
// close is forced from where any natural prefix of it stands (the other endings are never forced). Forced tokens are
// generated output (the total limit and usage count them); the budget never extends the total limit or the context,
// so a request that runs out of room during the close ends with its length limit and the reasoning unclosed.
// Reopening reasoning later in the turn counts against the same budget.
//
// UTF-8 (IFR1 R3). With config::utf8 the tracker follows the reasoning's bytes. When the budget is spent inside a
// character, the next positions stay free until the character's bytes are complete, at most max_extra tokens (a
// character lacks at most 3 continuation bytes; a token that cannot continue it, or a malformed one, ends the wait),
// and the close is forced after them. These extra tokens are reasoning but not counted against the budget (extra_n).
// If the wait ends without a complete character, the bytes stay as the model wrote them (the server shows the broken
// character as U+FFFD).
//
// Budget message (FBH1). The model goes on reasoning in its answer when the close alone cuts it off mid-thought, so
// config::message is fed before the forced close, the same way: forced tokens through the model. It is reserved inside
// the budget: with budget >= 4 x its tokens (reserve_ok) the free reasoning stops at budget - its tokens, then the
// message and the close follow; its tokens are reasoning and counted, so counted() <= budget still holds. A smaller
// budget has no message. The UTF-8 wait comes before the message, never inside or after it. The message is fed whole
// from its first token whatever the model wrote before it, except that tokens of the model's own already beginning
// the close at that point are completed into the close alone. Once fed, the close follows at once (and at once again
// if reasoning reopens).
//
// Sentence end (FBH1 L1). A message that cuts a sentence in half reads as an interruption, and the model picks the
// sentence up again in its answer. So, with a message reserved and config::utf8 given (it lists the tokens that end a
// sentence too, class_sentence), over the last wait_window(budget) = min(64, budget / 4) free tokens before the
// reserve the message starts right after the first token that ends a sentence; with none, at the reserve as before.
// The message only starts earlier, never later, so counted() <= budget holds. A request without a message closes at
// its budget as before.
//
// The forced close's form (FBH1 K1). The model writes its own close as a newline, the closing sequence, then a blank
// line, and after a close forced without them it goes on reasoning in its answer. So a forced close is config::lead
// (the newline), the closing sequence, then config::tail (the blank line), each a forced token through the model, at
// every budget and with or without a message. The lead is reasoning: it is reserved inside the budget like the message
// (the free reasoning stops at budget - its tokens, less the message's when one is reserved) and counted, so
// counted() <= budget still holds. With no room left in the budget (budget 0, or reasoning reopened once the budget is
// spent) it is fed all the same and, like the closing sequence, not counted. Lead tokens the reasoning already ends
// with (the model's own newline as its last free token, or a message's last tokens) are not fed twice, and a closing
// sequence of the model's own under way is completed alone, as before. The tail follows the close's last token when
// the budget forced that token: it is outside the reasoning and never counted. The model's own close is found as
// before and gets no tail (the model writes its own).
//
// Without a budget (FBH1 A3) a config that names the closing sequence is still followed: its reasoning tokens are
// counted, for the usage report, and nothing is ever forced.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fn::think {

struct config {
    int32_t budget = -1;                    // reasoning tokens the turn may generate; -1: unlimited (nothing tracked)
    std::vector<int32_t> open;              // the opening sequence (may be empty: then reasoning opens only at start)
    std::vector<int32_t> close;             // the closing sequence (required with a budget), forced when it is spent
    std::vector<std::vector<int32_t>> ends; // other sequences its parser ends reasoning with (natural, never forced)
    bool open_at_start = false;             // the prompt ends inside reasoning (a template that opens it)
    // IFR1 R3: utf8_entry of each token whose bytes are not whole UTF-8 characters, ascending by token (the server's
    // vocabulary); a token not listed is whole characters. Null: the bytes are not followed (the close is forced at the
    // budget). Shared and never changed, so a copy of the tracker is cheap. FBH1 L1: the tokens that end a sentence
    // are listed too (class_sentence)
    std::shared_ptr<const std::vector<uint32_t>> utf8;
    // FBH1: the budget message's tokens, fed before the forced close (empty: the close alone)
    std::vector<int32_t> message;
    // FBH1 K1: the tokens a forced close begins with (the model's newline before its closing sequence; reasoning,
    // reserved inside the budget) and ends with (its blank line after it; not reasoning). Empty: the sequence alone
    std::vector<int32_t> lead, tail;
};

constexpr int32_t max_extra = 3; // the most extra tokens a spent budget waits for a character's bytes
constexpr int32_t max_message = 256; // the most tokens a budget message may have

constexpr uint8_t class_sentence = 128; // FBH1 L1: a table entry's class bit for a token that ends a sentence

// FBH1 L1: the free tokens before the message's reserve in which the message starts at a sentence's end
inline int32_t wait_window(int32_t budget) {
    return budget / 4 < 64 ? budget / 4 : 64;
}

// FBH1: a budget of at least 4 x the message's tokens reserves it (budget 0 and tiny budgets close alone)
inline bool reserve_ok(int32_t budget, size_t message_n) {
    return message_n > 0 && static_cast<int64_t>(budget) >= 4 * static_cast<int64_t>(message_n);
}

// IFR1 R3: a token's bytes as UTF-8 (bits 0-2: the leading continuation bytes, 7 for 7 or more; 3-4: the continuation
// bytes its last character still lacks; 5: no byte after the leading continuation bytes; 6: a malformed sequence after
// them). 0: whole characters
uint8_t utf8_class(const char * bytes, size_t n);
// FBH1 L1: the token's text ends a sentence: its last character, spaces aside, is . ! ? (ASCII, or the CJK full
// stop, exclamation or question mark), or it ends with a newline (a line of notes is a sentence here)
bool sentence_end(const char * bytes, size_t n);
// the table entry for a token (below 2^24) and its bytes: its id and its class (utf8_class, with class_sentence when
// it ends a sentence); 0 for whole characters that end no sentence (not listed)
uint32_t utf8_entry(int32_t token, const char * bytes, size_t n);

// false with err set: a budget below -1, a budget without a closing sequence, a sequence empty (an ending) or longer
// than 16 tokens, a UTF-8 table not ascending by token, a message longer than max_message, holding an ending, or
// ending with tokens that begin the closing sequence (the close would swallow them), a lead holding a token of the
// closing sequence or of another ending, or a tail holding a token of the opening sequence
bool check(const config & c, std::string & err);

class tracker {
  public:
    void reset(const config & c);

    bool active() const { return c_.budget >= 0; }

    // the reasoning is followed and counted: a budget, or the closing sequence alone (FBH1 A3)
    bool tracks() const { return !c_.close.empty(); }

    bool reasoning() const { return in_; }

    // the token the next emitted position must be (the next one of the message, of the forced close's lead, of the
    // closing sequence or, after it, of the forced close's tail), -1: free
    int32_t forced() const;
    // the most tokens the next step may emit (>= 1) when its first n emitted positions are the proposals props (all
    // accepted); 1 when its first position is forced. A rejection earlier keeps every emitted position legal: the
    // positions before it are the same, and a free position stays free
    int32_t cap(const int32_t * props, int32_t n) const;
    // inside reasoning, the positions before the first one the budget can force (each emitted token adds at most one
    // to the count and the waiting tokens together); outside it, INT32_MAX (0 while a forced close's tail is due)
    int32_t room() const;
    // one emitted token; returns the reasoning tokens it added to the count (0 for content and sequence tokens; more
    // than 1 when it ends a false prefix of the closing sequence)
    int32_t observe(int32_t token);

    int32_t counted() const { return body_; }  // reasoning tokens counted so far
    int32_t forced_n() const { return n_forced_; }  // message, lead, closing and tail tokens forced so far
    int32_t extra_n() const { return n_extra_; }    // reasoning tokens past the budget, finishing a character
    int32_t budget() const { return c_.budget; }

  private:
    // what a forced token belongs to
    enum class part : uint8_t { none, message, lead, close, tail };

    bool waiting() const; // the budget is spent and the next position waits for a character's bytes
    // the count that ends the free reasoning: the budget, less the forced close's lead and a message still to feed
    int32_t limit() const;
    int32_t window() const; // the sentence-end window before limit() (0 without a reserved message or a table)
    bool early() const;     // inside the window right after a token that ended a sentence: the message starts here
    int32_t forced_from(part & p) const; // forced(), and what the token belongs to
    uint8_t class_of(int32_t token) const;

    config c_;
    bool in_ = false;
    int32_t body_ = 0;          // counted reasoning tokens
    std::vector<int32_t> pend_; // inside reasoning: the last tokens, not yet counted, that begin an ending
    int32_t om_ = 0;            // the opening sequence's prefix the last tokens match (outside reasoning)
    int32_t n_forced_ = 0;
    int32_t need_ = 0;    // the continuation bytes the reasoning's last character lacks (with config::utf8)
    int32_t waited_ = 0;  // positions taken while the spent budget waited for a character
    bool broken_ = false; // such a position's token could not continue the character
    int32_t n_extra_ = 0;
    bool msg_on_ = false; // the budget reserves the message
    bool sent_ = false;   // the last reasoning token was free and ended a sentence
    int32_t msg_ = 0;     // message tokens fed this turn
    int32_t lead_m_ = 0;  // the lead's prefix the reasoning's last tokens are (K1)
    bool closing_ = false; // a forced close's lead is under way or fed: every position up to the close is forced
    int32_t tail_ = -1;   // outside reasoning after a forced close: the tail tokens fed so far; -1: no tail due
};

} // namespace fn::think
