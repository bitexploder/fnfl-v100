// TTTSBS: fnfl-safety, the host safety build (no CUDA, no card): the lifetime ledger (fn/xchg/life.h) and the link's
// test hooks (fn/xchg/link_fault.h) compiled in, under ASan + UBSan or TSan when its build directory says so.
//   fnfl-safety safety-test [all|CASES]   the split's buffer lifetimes on the failure paths (fn/runtime/safety_test.h)
//   fnfl-safety link-test leader|follower [--addr IP] [--port P] [--timeout S] [--cases LIST] [--stream N]
//                                          the link-test's host-buffer mode (fn/xchg/link_test.h, --host implied)
//   fnfl-safety pipe-test                  the two-group pipeline core (fn/runtime/pipe.h), as fnfl-drive's

#include "fn/runtime/pipe.h"
#include "fn/runtime/safety_test.h"
#include "fn/xchg/link_test.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char ** argv) {
    if (argc >= 2 && strcmp(argv[1], "safety-test") == 0) {
        return fn::runtime::fnfl_safety_test(argc >= 3 ? argv[2] : "all");
    }
    if (argc >= 2 && strcmp(argv[1], "pipe-test") == 0) {
        std::string report;
        const bool  ok = fn::runtime::pipe::self_test(report);
        fputs(report.c_str(), stdout);
        return ok ? 0 : 1;
    }
    if (argc >= 3 && strcmp(argv[1], "link-test") == 0 &&
        (strcmp(argv[2], "leader") == 0 || strcmp(argv[2], "follower") == 0)) {
        fn::xchg::link::test_options o;
        o.leader = strcmp(argv[2], "leader") == 0;
        o.host   = true;
        o.addr   = "127.0.0.1";
        bool ok  = true;
        for (int i = 3; ok && i < argc; ++i) {
            const bool has = i + 1 < argc;
            if (strcmp(argv[i], "--addr") == 0 && has) {
                o.addr = argv[++i];
            } else if (strcmp(argv[i], "--port") == 0 && has) {
                o.port = static_cast<uint16_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "--timeout") == 0 && has) {
                o.timeout_s = atof(argv[++i]);
            } else if (strcmp(argv[i], "--cases") == 0 && has) {
                o.cases = argv[++i];
            } else if (strcmp(argv[i], "--stream") == 0 && has) {
                o.stream_messages = atol(argv[++i]);
            } else if (strcmp(argv[i], "--host") == 0) {
            } else {
                ok = false;
            }
        }
        if (ok && o.timeout_s > 0 && o.stream_messages >= 1000) {
            return fn::xchg::link::run_test(o);
        }
    }
    fprintf(stderr,
            "usage: fnfl-safety safety-test [all|CASES] | pipe-test | link-test leader|follower [--addr IP] "
            "[--port P] [--timeout S] [--cases LIST] [--stream N >= 1000]\n");
    return 2;
}
