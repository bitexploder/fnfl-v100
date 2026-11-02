#pragma once

// TTTSBS: the lifetime ledger of the split's host buffers, a test build's diagnostic (-DFNFL_LIFETIME: the host
// safety tests, or an instrumented drive). In every other build each call below is an empty inline, so serving pays
// nothing. The production ownership paths (fn/xchg/link.cpp, fn/runtime/stage_wire.cpp, fn/runtime/stage.cpp) call it
// at each boundary of a buffer's life:
//   own / disown    an owner's storage begins / is destroyed (a record allocated, pinned memory freed);
//   map / unmap     its registration with the transport (ucp_mem_map / ucp_mem_unmap);
//   use / done      an asynchronous user begins / is observed ended: a transport operation (submitted / its end
//                   acquired by the runtime, a completion or an acknowledged cancellation), or a GPU copy (queued /
//                   its completion record acquired);
//   reuse           the owner refills or repurposes the storage (the next round's boundary, the next crossing);
//   publish         a transport writes a completion's bytes into the storage (the test transport's delayed copy).
// disown, unmap and reuse assert, at that boundary, that no user of the range remains; done asserts the user is known
// (a stale handle cannot end a newer user); publish asserts the owner is alive and still the generation the user
// started on (a stale completion cannot land in a new occupant). A violation prints one "TTTSBS LIFETIME:" line and
// aborts, or (set_abort(false), the tests' demonstration of the old paths) is counted.

#include <cstddef>
#include <cstdint>
#include <string>

namespace fn::life {

struct totals {
    int64_t owns = 0, disowns = 0, maps = 0, unmaps = 0, uses = 0, dones = 0, reuses = 0, publishes = 0;
    int64_t violations  = 0;
    int64_t live_owners = 0, live_users = 0;
};

#if defined(FNFL_LIFETIME)
constexpr bool on = true;
void           own(const void * p, size_t n, const char * what);
void           disown(const void * p, const char * what);
void           map(const void * p, size_t n);
void           unmap(const void * p, size_t n);
uint64_t       next_id();
void           use(uint64_t user, const void * p, size_t n, const char * what);
void           done(uint64_t user, const char * how);
void           reuse(const void * p, size_t n, const char * what);
void           publish(uint64_t user);
// abort on a violation (true, the default) or count it
void           set_abort(bool a);
// one stderr line per event (also FNFL_LIFE_LOG=1)
void           set_log(bool l);
totals         get();
// the last violation's line ("" when none)
std::string    last();
// counts and the last violation cleared (live owners and users stay)
void           reset_counts();
#else
constexpr bool on = false;

inline void own(const void *, size_t, const char *) {}

inline void disown(const void *, const char *) {}

inline void map(const void *, size_t) {}

inline void unmap(const void *, size_t) {}

inline uint64_t next_id() {
    return 0;
}

inline void use(uint64_t, const void *, size_t, const char *) {}

inline void done(uint64_t, const char *) {}

inline void reuse(const void *, size_t, const char *) {}

inline void publish(uint64_t) {}

inline void set_abort(bool) {}

inline void set_log(bool) {}

inline totals get() {
    return totals();
}

inline std::string last() {
    return std::string();
}

inline void reset_counts() {}
#endif

}  // namespace fn::life
