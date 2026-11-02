#pragma once

// MU1: the two-user scheduling policy's pure parts, ported from Superfast's TU1..TU6 (Qwen27B main a2a272f,
// tools/server/server-context.cpp: tu_order :4574, tu_slice :4701, tu_remaining :4552, get_available_slot :2085), with
// no server or engine type, so tools/fnfl-splice-test checks them without a model. TU6's fix and fair_read are FNFL's
// permanent behavior (STANDARDS "Switches": a decided switch keeps one path).

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <system_error>
#include <vector>

// one reader as tu_order ranks it
struct tu_reader {
    int     slot      = -1;
    int32_t priority  = 0;
    int32_t remaining = 0;  // prompt tokens left to read
    int32_t uncached  = 0;  // the read's uncached length (its short/long class)
    int64_t last_seq  = 0;  // tu_last_read_seq: when its last turn began
    int     id_task   = -1;
    bool    live      = false; // its chunk is in progress (the live job's reader)
    bool begun = false;        // FMU1: its read job is prepared (fnfl_read_begin ran)
    bool    finishing = false;        // PRD1 item 9: its front ended, its back halves are out (polled on its own)
};

// TU6's order: priority (higher first); short readers (uncached <= fast_limit) before long ones; among long readers the
// live job's reader keeps its turn, then the earlier last turn, then the lower task ID; among short readers fewer
// remaining first, then the lower task ID
inline bool tu_before(const tu_reader & a, const tu_reader & b, int32_t fast_limit) {
    if (a.priority != b.priority) {
        return a.priority > b.priority;
    }
    const bool a_short = a.uncached <= fast_limit;
    const bool b_short = b.uncached <= fast_limit;
    if (a_short != b_short) {
        return a_short;
    }
    if (!a_short) {
        if (a.live != b.live) {
            return a.live;
        }
        if (a.last_seq != b.last_seq) {
            return a.last_seq < b.last_seq;
        }
        return a.id_task < b.id_task;
    }
    if (a.remaining != b.remaining) {
        return a.remaining < b.remaining;
    }
    return a.id_task < b.id_task;
}

inline void tu_sort(std::vector<tu_reader> & r, int32_t fast_limit) {
    std::sort(r.begin(), r.end(), [&](const tu_reader & a, const tu_reader & b) { return tu_before(a, b, fast_limit); });
}

// PRD1 item 9: the readers TU6's order ranks: a finishing reader (its front done, 0 rows left, so it would sort first)
// leaves the order, the pick and the reader to follow
inline void tu_drop_finishing(std::vector<tu_reader> & r) {
    r.erase(std::remove_if(r.begin(), r.end(), [](const tu_reader & x) { return x.finishing; }), r.end());
}

// the reader to serve next: the first in TU6's order of the readers not finishing (-1: none)
inline int tu_pick_of(std::vector<tu_reader> r, int32_t fast_limit) {
    tu_drop_finishing(r);
    tu_sort(r, fast_limit);
    return r.empty() ? -1 : r[0].slot;
}

// TU2/TU3's remaining tokens. A read not yet begun counts its prompt less the prefix its begin will reuse (computed
// once per task); a begun read counts its prompt less the tokens its slot's state holds (FNFL's job covers the whole
// read, so its unfinished rows are counted once). uncached: the larger of that and the prompt less the reused prefix
inline int32_t tu_remaining_tokens(bool begun, int32_t n_tokens, int32_t prefix, int32_t committed) {
    return std::max(0, n_tokens - (begun ? committed : prefix));
}
inline int32_t tu_uncached(int32_t remaining, int32_t n_tokens, int32_t prefix) {
    return std::max(remaining, n_tokens - prefix);
}

// FMU1: the reader TU6's order serves after the current main-lane turn (now_slot's, -1: none) ends: the order with no
// live reader and now's turn the latest; the first reader whose job is begun (-1: none), possibly now itself
inline int tu_follow(std::vector<tu_reader> r, int now_slot, int32_t fast_limit) {
    tu_drop_finishing(r);
    int64_t latest = 0;
    for (const tu_reader & x : r) {
        latest = std::max(latest, x.last_seq);
    }
    for (tu_reader & x : r) {
        x.live = false;
        if (x.slot == now_slot) {
            x.last_seq = latest + 1;
        }
    }
    tu_sort(r, fast_limit);
    for (const tu_reader & x : r) {
        if (x.begun) {
            return x.slot;
        }
    }
    return -1;
}

// FMU1: the presence rule (resident TP). What a reader's step runs beside: nobody else with model work (lone: FMU2's
// correction, TU3's geometry, one step a job call, so an arrival waits one step, not the chunk), generating slots
// (writers: one step a slice, its units under the read budget, MUP2; APF1: the writer-pause deadline's), or other readers only (shared: TU3's slices of
// the policy's groups)
// PRD1 item 14: what the shell does at a read's end (the engine's done, cancelled or failed; cancel: the client asked
// for one during the read). A cancel is honoured at any point of the read: a read that ended done while a cancel was
// pending (the split's front was done before it came) releases the slot like a cancelled one, with no generated token
enum class tu_read_out { done, cancelled, failed };
enum class tu_end { generate, release, fail };

inline tu_end tu_read_end(tu_read_out st, bool cancel) {
    if (st == tu_read_out::failed) {
        return tu_end::fail;
    }
    return st == tu_read_out::done && !cancel ? tu_end::generate : tu_end::release;
}

enum class tu_mode { shared, writers, lone };
inline tu_mode tu_presence(int32_t n_writers, int32_t n_other_readers) {
    return n_writers > 0 ? tu_mode::writers : n_other_readers > 0 ? tu_mode::shared : tu_mode::lone;
}
// the slice budget a mode's steps run in (lone: one step, the whole chunk). FMU2 (the reader rule, Fable review 1
// section 4): writers too have the whole slice, main's, not FMU1's half
inline double tu_mode_budget(tu_mode m, double slice_ms) {
    (void)m;
    return slice_ms;
}

// APF1: the writers' pause target P (POST /props tu.writer_pause_ms, --fnfl-writer-pause-ms): an integer from 10 to
// 1,000 ms (fn/runtime/pace.h's range; the engine checks it again). A rejected value changes nothing
constexpr int32_t tu_pause_min_ms = 10;
constexpr int32_t tu_pause_max_ms = 1000;

inline bool tu_pause_ok(bool is_integer, int64_t v) {
    return is_integer && v >= tu_pause_min_ms && v <= tu_pause_max_ms;
}

// POST /props tu.writer_pause_ms (a JSON value J, nlohmann's): true with out set when valid; else false with err and out
// untouched (the request is rejected whole, so the policy's P stays)
template <typename J> inline bool tu_pause_parse(const J & v, int32_t & out, std::string & err) {
    if (!v.is_number_integer() || !tu_pause_ok(true, v.template get<int64_t>())) {
        err = "tu.writer_pause_ms must be an integer from 10 to 1000";
        return false;
    }
    out = v.template get<int32_t>();
    return true;
}

// FMU1: a lone writer (one generating slot, no reader, admission or job) runs the one-slot loop's body
inline bool tu_lone_writer(int32_t n_writers, int32_t n_readers, bool job_active) {
    return n_writers == 1 && n_readers == 0 && !job_active;
}
// FMU1: admissions an iteration: one while any slot has model work (the next one after the writers' rounds), else all
inline bool tu_admit_more(bool busy, int32_t n_admitted) {
    return !busy || n_admitted == 0;
}

// TU3's one-sync slice: the groups of a slice's next step. budget_ms the slice, t_slice_ms its time so far, n_run its
// steps so far, last_group_ms the chunk's last measured group (-1: none, as at a chunk's start): with an estimate,
// floor((budget - t_slice) / g) groups (at least one, at most 1,024), ending the slice (0) once a step ran and none
// fits; without one, one group, measured, then the slice ends
inline int32_t tu_next_groups(double budget_ms, double t_slice_ms, double last_group_ms, int32_t n_run) {
    if (last_group_ms >= 0.0) {
        const double g = std::max(last_group_ms, 1e-3);
        const double n = std::floor((budget_ms - t_slice_ms) / g);
        if (n_run > 0 && n < 1.0) {
            return 0;
        }
        return static_cast<int32_t>(std::min(1024.0, std::max(1.0, n)));
    }
    return n_run > 0 ? 0 : 1;
}

// TU6's least recently used available slot: idle ties go to the lowest slot (<, not <=); -1 when none is available
struct tu_slot_view {
    bool    available   = false;
    int64_t t_last_used = -1;
};
inline int tu_lru(const std::vector<tu_slot_view> & s) {
    int     ret    = -1;
    int64_t t_last = -1;
    for (int i = 0; i < static_cast<int>(s.size()); ++i) {
        if (!s[size_t(i)].available) {
            continue;
        }
        if (ret < 0 || s[size_t(i)].t_last_used < t_last) {
            t_last = s[size_t(i)].t_last_used;
            ret    = i;
        }
    }
    return ret;
}

// SSF1: a client's slot ID names a slot, [0, n_slots), or is exactly -1 where "any slot" is allowed (a request's
// id_slot). Upstream wrapped any integer (id % n_slots, unsigned), so 99 and -1 named real slots
inline bool slot_id_valid(int64_t id, int32_t n_slots, bool allow_any) {
    return (id >= 0 && id < n_slots) || (allow_any && id == -1);
}

// SSF1: /slots/{id}'s path ID: the whole string is a decimal integer (no prefix, sign other than '-', or blank) that
// names a slot; false otherwise, with id untouched
inline bool slot_id_parse(const std::string & s, int32_t n_slots, int32_t & id) {
    int64_t                      v   = 0;
    const char *                 end = s.data() + s.size();
    const std::from_chars_result r   = std::from_chars(s.data(), end, v);
    if (s.empty() || r.ec != std::errc() || r.ptr != end || !slot_id_valid(v, n_slots, false)) {
        return false;
    }
    id = static_cast<int32_t>(v);
    return true;
}
