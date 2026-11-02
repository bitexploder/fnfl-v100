#include "fn/xchg/life.h"

#if defined(FNFL_LIFETIME)

#    include <atomic>
#    include <cstdio>
#    include <cstdlib>
#    include <map>
#    include <mutex>

namespace fn::life {

namespace {

struct owner {
    size_t      n      = 0;
    uint32_t    gen    = 0;
    int32_t     mapped = 0;
    std::string what;
};

struct user {
    uintptr_t   p = 0, n = 0;
    uintptr_t   at  = 0;  // the owner's start (0: storage no owner declared)
    uint32_t    gen = 0;  // the owner's generation when the use began
    std::string what;
};

struct ledger {
    std::mutex                 mu;
    std::map<uintptr_t, owner> owners;
    std::map<uint64_t, user>   users;
    std::atomic<uint64_t>      ids{ 1 };
    bool                       abort_on = true;
    bool                       log      = getenv("FNFL_LIFE_LOG") != nullptr;
    totals                     t;
    std::string                last;
};

ledger & L() {
    static ledger l;
    return l;
}

uintptr_t u(const void * p) {
    return reinterpret_cast<uintptr_t>(p);  // an address compared, never dereferenced
}

// the owner whose range holds p (nullptr: none)
std::map<uintptr_t, owner>::iterator owner_of(ledger & l, uintptr_t p) {
    auto it = l.owners.upper_bound(p);
    if (it == l.owners.begin()) {
        return l.owners.end();
    }
    --it;
    return p < it->first + it->second.n ? it : l.owners.end();
}

void violate(ledger & l, const std::string & what) {
    ++l.t.violations;
    l.last = "TTTSBS LIFETIME: " + what;
    fprintf(stderr, "%s\n", l.last.c_str());
    fflush(stderr);
    if (l.abort_on) {
        abort();
    }
}

void say(ledger & l, const std::string & what) {
    if (l.log) {
        fprintf(stderr, "TTTSBS life: %s\n", what.c_str());
    }
}

std::string hex(uintptr_t p) {
    char b[32];
    snprintf(b, sizeof b, "%#llx", (unsigned long long) p);
    return b;
}

// every user overlapping [p, p + n): a violation naming the boundary
void no_users(ledger & l, uintptr_t p, size_t n, const std::string & boundary) {
    for (const auto & [id, x] : l.users) {
        if (x.p < p + n && p < x.p + x.n) {
            violate(l, boundary + " at " + hex(p) + " (" + std::to_string(n) + " bytes) while user " +
                           std::to_string(id) + " (" + x.what + ", " + std::to_string(x.n) + " bytes at " + hex(x.p) +
                           ") has not ended");
        }
    }
}

}  // namespace

void own(const void * p, size_t n, const char * what) {
    ledger &                    l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.t.owns;
    owner & o = l.owners[u(p)];
    o         = owner();
    o.n       = n;
    o.what    = what;
    say(l, std::string("own ") + what + " " + hex(u(p)) + " " + std::to_string(n));
}

void disown(const void * p, const char * what) {
    ledger &                    l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.t.disowns;
    auto it = l.owners.find(u(p));
    if (it == l.owners.end()) {
        violate(l, std::string("disown of ") + what + " at " + hex(u(p)) + ": no such owner");
        return;
    }
    no_users(l, it->first, it->second.n, std::string("destruction of ") + what);
    if (it->second.mapped > 0) {
        violate(l, std::string("destruction of ") + what + " at " + hex(u(p)) + " while it is still mapped");
    }
    say(l, std::string("disown ") + what + " " + hex(u(p)));
    l.owners.erase(it);
}

void map(const void * p, size_t n) {
    ledger &                    l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.t.maps;
    auto it = owner_of(l, u(p));
    if (it != l.owners.end()) {
        ++it->second.mapped;
    }
    say(l, "map " + hex(u(p)) + " " + std::to_string(n));
}

void unmap(const void * p, size_t n) {
    ledger &                    l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.t.unmaps;
    no_users(l, u(p), n, "unmapping");
    auto it = owner_of(l, u(p));
    if (it != l.owners.end() && it->second.mapped > 0) {
        --it->second.mapped;
    }
    say(l, "unmap " + hex(u(p)) + " " + std::to_string(n));
}

uint64_t next_id() {
    return L().ids.fetch_add(1, std::memory_order_relaxed);
}

void use(uint64_t id, const void * p, size_t n, const char * what) {
    ledger &                    l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.t.uses;
    user x;
    x.p     = u(p);
    x.n     = n;
    x.what  = what;
    auto it = owner_of(l, u(p));
    if (it != l.owners.end()) {
        x.at  = it->first;
        x.gen = it->second.gen;
    }
    if (!l.users.emplace(id, x).second) {
        violate(l, "user " + std::to_string(id) + " began twice (" + what + ")");
    }
    say(l, "use " + std::to_string(id) + " " + what + " " + hex(u(p)) + " " + std::to_string(n));
}

void done(uint64_t id, const char * how) {
    ledger &                    l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.t.dones;
    auto it = l.users.find(id);
    if (it == l.users.end()) {
        violate(l, "user " + std::to_string(id) + " ended (" + how + ") but is not outstanding: a stale handle");
        return;
    }
    say(l, "done " + std::to_string(id) + " " + it->second.what + " (" + how + ")");
    l.users.erase(it);
}

void reuse(const void * p, size_t n, const char * what) {
    ledger &                    l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.t.reuses;
    no_users(l, u(p), n, std::string("reuse of ") + what);
    auto it = owner_of(l, u(p));
    if (it != l.owners.end()) {
        ++it->second.gen;
    }
    say(l, std::string("reuse ") + what + " " + hex(u(p)) + " " + std::to_string(n));
}

void publish(uint64_t id) {
    ledger &                    l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.t.publishes;
    auto it = l.users.find(id);
    if (it == l.users.end()) {
        violate(l, "a completion of user " + std::to_string(id) + " published after the user ended");
        return;
    }
    const user & x = it->second;
    if (x.at != 0) {
        auto o = l.owners.find(x.at);
        if (o == l.owners.end()) {
            violate(l, "a completion of user " + std::to_string(id) + " (" + x.what + ") published into " + hex(x.p) +
                           " after its owner was destroyed");
        } else if (o->second.gen != x.gen) {
            violate(l, "a completion of user " + std::to_string(id) + " (" + x.what + ") published into " + hex(x.p) +
                           " of generation " + std::to_string(o->second.gen) + ", begun on generation " +
                           std::to_string(x.gen) + ": a stale completion into a new occupant");
        }
    }
}

void set_abort(bool a) {
    ledger &                    l = L();
    std::lock_guard<std::mutex> g(l.mu);
    l.abort_on = a;
}

void set_log(bool on_) {
    ledger &                    l = L();
    std::lock_guard<std::mutex> g(l.mu);
    l.log = on_;
}

totals get() {
    ledger &                    l = L();
    std::lock_guard<std::mutex> g(l.mu);
    totals                      t = l.t;
    t.live_owners                 = int64_t(l.owners.size());
    t.live_users                  = int64_t(l.users.size());
    return t;
}

std::string last() {
    ledger &                    l = L();
    std::lock_guard<std::mutex> g(l.mu);
    return l.last;
}

void reset_counts() {
    ledger &                    l = L();
    std::lock_guard<std::mutex> g(l.mu);
    l.t    = totals();
    l.last = std::string();
}

}  // namespace fn::life

#endif
