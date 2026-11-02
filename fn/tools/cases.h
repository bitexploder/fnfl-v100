#pragma once

// TFC: FNFL's check inventory. Every check and optional tool of fnfl-drive, fnfl-safety, fnfl-splice-test and the
// curated serving suite, with a stable ID, what it needs (host, host with data files, cards, a model load, a server),
// its fixtures and its exact arguments. Listing reads nothing but this table and the fixtures' existence: no device,
// no model, no download, no lock.
//   fnfl-drive cases [FILTER]                  the table (FILTER: a group such as host or gpu, or an ID prefix)
//   fnfl-drive cases fixtures                  each fixture: name, environment override, path, present or missing
//   fnfl-drive cases run [FILTER] [--out DIR] [--timeout S]
//                                              the host checks only (host and host-data), each as a child process with
//                                              CUDA_VISIBLE_DEVICES empty, its output in DIR/<ID>.log. A check whose
//                                              fixture or tool is missing is NOT RUN and is not a pass. Prints one line
//                                              a check and a summary; exit 0 when every selected check passed, 1 when
//                                              any failed, 3 when none failed but some were not run. A GPU, model or
//                                              serving check is never run here: its command needs the ticket's locks.

#include <cstdint>
#include <string>

namespace fn::cases {

// what a check needs. host: the CPU only; host_data: the CPU and fixture files (a GGUF read on the CPU, a dump);
// gpu: cards, synthetic data; gpu_data: cards and fixture files, no model load (zero weights or tensors read on the
// CPU); model: a model load onto the cards; serving: a server process with the model; two_box: both boxes
enum class need : uint8_t { host, host_data, gpu, gpu_data, model, serving, two_box };

// check: exits 0 for PASS; tool: an optional bench, probe, prototype or fixture generator (not a check)
enum class kind : uint8_t { check, tool };

struct fixture {
    const char * name;
    const char * env;   // the environment variable that overrides the path
    const char * path;  // the default; empty: none, the variable is required
    const char * what;
};

struct entry {
    const char * id;  // stable
    kind         k;
    need         n;
    int32_t      cards;   // the cards used (0, 1 or 2); a GPU process holds all three canonical locks
    const char * tool;    // fnfl-drive, fnfl-bench, fnfl-safety, fnfl-splice-test or pytest
    const char * args;    // {name}: the named fixture's path; {out}: the check's own output directory
    const char * peer;    // a second process run beside the first (link-test's follower), or empty
    const char * covers;  // one line
};

// the fixture's path: its environment variable when set, else its default; empty when neither
std::string fixture_path(const std::string & name);

const char * need_name(need n);

int list(const std::string & filter);
int list_fixtures();
int run(const char * self, const std::string & filter, const std::string & out_dir, int32_t timeout_s);

// fnfl-drive cases ...: argv[2..]
int main(int argc, char ** argv);

}  // namespace fn::cases
