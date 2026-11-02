#pragma once

// SSF1: the slow-wait trace, permanent in serving. About once in 25 minutes of DeepSWE every writer of the split stops
// for 1.00-1.09 s (BACKLOG, "Split loop: a ~1 s whole-pipeline stall"); these lines name where. A wait, turn or
// command of threshold_s or more prints one line, and only while at least one writer generates on this box (on() is
// the stall's own definition): an idle follower's 1 s command waits and a lone read's long chunks print nothing. The
// cost on a normal round is the clock read the caller already takes and one compare.
//
// A line: "FNFL slow: <what> <s> s[, <detail>] (UTC HH:MM:SS.uuuuuu, t_us <monotonic>)". The UTC wall clock joins
// the two boxes' logs (the follower's has no timestamps of its own; the boxes' time zones differ); t_us is
// CLOCK_MONOTONIC, as the server's "TU iteration ... t_us=" lines print it; a link wait names its tag (kind, group,
// sequence, card, aux), so leader and follower lines join by sequence. On the leader the server sets the sink to its
// log, so each line carries the log's own timestamp as well (what ADHOC-DS13's stall_scan.py reads).

#include <cstdint>
#include <string>

namespace fn::xchg::slow {

constexpr double threshold_s = 0.5;

// the slots generating on this box, a bit a slot (slots 0 .. 63): the leader's server sets the whole mask each turn,
// the follower a slot's bit as it serves the slot's step groups and clears it at the slot's next request
void set_writers(uint64_t mask);
void writer(int32_t slot, bool on);
// at least one writer generates on this box
bool on();
// true when a wait of s seconds is a line (s >= threshold_s while on())
bool due(double s);

// where lines go: sink(line) (the line has no newline); nullptr (the default): stderr
void set_sink(void (*sink)(const char * line));
// one line (unconditional: the caller checked due)
void line(const std::string & what, double s, const std::string & detail = std::string());

}  // namespace fn::xchg::slow
