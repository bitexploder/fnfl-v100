#include "fn/kernels/experts/grids_test.h"

#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/grids.h"
#include "fn/loader/sha256.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ggml/src/ggml-common.h at 66e11c3
constexpr const char * pinned_sha256 = "9e47bb2e1e776341f70719767f490d00962cfc6029e1b5721afc157ce9be7050";

// the values of GGML_TABLE_BEGIN(type, name, n) .. GGML_TABLE_END() as little-endian bytes of type; empty when the
// table is missing or its type or count differ
std::vector<uint8_t> table_bytes(const std::string & text, const char * type, const char * name, int32_t n,
                                 int32_t & first_line) {
    const std::string begin = std::string("GGML_TABLE_BEGIN(") + type + ", " + name + ", " + std::to_string(n) + ")";
    const size_t at = text.find(begin);
    const size_t end = at == std::string::npos ? at : text.find("GGML_TABLE_END()", at);
    if (end == std::string::npos) {
        return {};
    }
    first_line = static_cast<int32_t>(std::count(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(at), '\n')) + 2;
    const size_t body = text.find('\n', at) + 1;
    const std::string vals = text.substr(body, end - body);
    const size_t width = std::strcmp(type, "uint64_t") == 0 ? 8 : std::strcmp(type, "uint32_t") == 0 ? 4 : 1;
    static const std::regex number(R"(-?0x[0-9a-fA-F]+|-?[0-9]+)");
    std::vector<uint8_t> out;
    for (auto it = std::sregex_iterator(vals.begin(), vals.end(), number); it != std::sregex_iterator(); ++it) {
        const std::string tok = it->str();
        // the low width bytes of the value; a negative int8_t entry is its two's complement
        const uint64_t v = tok[0] == '-' ? static_cast<uint64_t>(std::strtoll(tok.c_str(), nullptr, 0))
                                         : std::strtoull(tok.c_str(), nullptr, 0);
        for (size_t b = 0; b < width; ++b) {
            out.push_back(static_cast<uint8_t>(v >> (8 * b)));
        }
    }
    return out.size() == size_t(n) * width ? out : std::vector<uint8_t>();
}

} // namespace

int fnfl_grid_test(const char * header) {
    using namespace fn::experts;
    std::ifstream f(header, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    if (!f || text.empty()) {
        fprintf(stderr, "grid-test: cannot read %s\n", header);
        return 1;
    }
    fn::loader::sha256 h;
    h.update(text.data(), text.size());
    const std::string got = fn::loader::hex(h.finish());
    printf("grid-test: %s, sha256 %s (66e11c3: %s)\n", header, got.c_str(), got == pinned_sha256 ? "yes" : "NO");
    bool pass = got == pinned_sha256;
    const struct {
        const char * type;
        const char * name;
        int32_t n;
        const void * ours;
        size_t bytes;
    } tables[6] = {{"uint64_t", "iq2xxs_grid", 256, iq2xxs_grid, sizeof(iq2xxs_grid)},
                   {"uint64_t", "iq2xs_grid", 512, iq2xs_grid, sizeof(iq2xs_grid)},
                   {"uint64_t", "iq2s_grid", 1024, iq2s_grid, sizeof(iq2s_grid)},
                   {"uint32_t", "iq3xxs_grid", 256, iq3xxs_grid, sizeof(iq3xxs_grid)},
                   {"uint32_t", "iq3s_grid", 512, iq3s_grid, sizeof(iq3s_grid)},
                   {"int8_t", "kvalues_iq4nl", 16, kvalues_iq4nl, sizeof(kvalues_iq4nl)}};
    for (const auto & t : tables) {
        int32_t line = 0;
        const std::vector<uint8_t> def = table_bytes(text, t.type, t.name, t.n, line);
        size_t differ = def.size() == t.bytes ? 0 : t.bytes;
        for (size_t i = 0; differ == 0 && i < t.bytes; ++i) {
            differ += def[i] != static_cast<const uint8_t *>(t.ours)[i];
        }
        printf("grid-test: %-14s %4d x %s from line %d, %5zu bytes: %s\n", t.name, t.n, t.type, line, t.bytes,
               differ == 0 ? "identical" : "DIFFER");
        pass = pass && differ == 0;
    }
    std::vector<uint8_t> buf(grid_bytes);
    const bool packed = grid_table(grid_src, grid_src_bytes, buf.data());
    printf("grid-test: grid_table packs the six and the selector tables into %zu bytes: %s\n", grid_bytes,
           packed ? "yes" : "NO");
    pass = pass && packed;
    // FMS4 item 2: each selector region against its derivation here from the 8-byte table: entry e's nibble k is the
    // level code of byte k (8 -> 0, 25 -> 1, 43 -> 2)
    const struct {
        const char * name;
        const uint64_t * grid;
        int32_t n;
        size_t off;
    } sels[3] = {
        {"iq2xxs", iq2xxs_grid, 256, 17424}, {"iq2xs", iq2xs_grid, 512, 18448}, {"iq2s", iq2s_grid, 1024, 20496}};
    for (const auto & t : sels) {
        size_t differ = 0;
        for (int32_t e = 0; e < t.n; ++e) {
            uint32_t want = 0;
            for (int32_t k = 0; k < 8; ++k) {
                const uint32_t b = static_cast<uint32_t>(t.grid[e] >> (8 * k)) & 0xffu;
                want |= (b == 8 ? 0u : b == 25 ? 1u : b == 43 ? 2u : 0xfu) << (4 * k);
            }
            for (int32_t i = 0; i < 4; ++i) {
                const size_t at = t.off + 4 * static_cast<size_t>(e) + static_cast<size_t>(i);
                differ += buf[at] != ((want >> (8 * i)) & 0xffu);
            }
        }
        printf("grid-test: %-6s selector table at %5zu, %4d x 4 bytes: %s\n", t.name, t.off, t.n,
               differ == 0 ? "identical" : "DIFFER");
        pass = pass && differ == 0;
    }
    const int64_t wrong = packed ? fnfl_grid_sel_device(buf.data(), 0) : -1;
    printf("grid-test: the selector recipe on the device: %lld of 360448 cases wrong\n", static_cast<long long>(wrong));
    pass = pass && wrong == 0;
    printf("grid-test: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
