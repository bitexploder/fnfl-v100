#include "fn/tools/test_dump.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

namespace fn::tools {

bool test_dump(const std::string & label, const void * p, size_t n) {
    const char * dir = std::getenv("FNFL_TEST_DUMP");
    if (dir == nullptr || dir[0] == '\0') {
        return true;
    }
    static std::mutex mu;
    static std::map<std::string, int> seen;
    std::string name;
    for (const char c : label) { // a file name: letters, digits, '-', '.', '_' kept, everything else '_'
        const bool keep = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '-' ||
                          c == '.' || c == '_';
        name += keep ? c : '_';
    }
    int k = 0;
    {
        const std::lock_guard<std::mutex> lock(mu);
        k = seen[name]++;
    }
    const std::string path = std::string(dir) + "/" + name + "." + std::to_string(k) + ".bin";
    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "test_dump: cannot write %s\n", path.c_str());
        return false;
    }
    const bool ok = (n == 0 || std::fwrite(p, 1, n, f) == n);
    return std::fclose(f) == 0 && ok;
}

} // namespace fn::tools
