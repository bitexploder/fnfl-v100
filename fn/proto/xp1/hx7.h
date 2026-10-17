#pragma once

// XP1 Amendment 3: HX7's captured traffic at 58,581 cells (work/HX7/runs/capture-58581-20261009T090240Z, read only):
// three-row teacher-forced decodes of a real transcript, 1,024 decodes. Read once on the CPU (gzip through a pipe),
// then copied to device buffers that graph replays index by a device counter.
//   routes.txt.gz:    lines "decode row layer e0 .. e9" (row counts tokens globally: decode d holds rows 3d .. 3d + 2)
//   routes.txt.sel.*: layer 11's selected cells, 2,051 I32 a row (plane 0), rows in .sel.idx order ("decode row layer
//                     n", n the selected cells, padding excluded)

#include <cstdint>
#include <string>
#include <vector>

namespace fn::xp1 {

constexpr int32_t hx7_rows = 3;
constexpr int32_t hx7_width = 2051;

struct hx7_traffic {
    int32_t n_decode = 0;
    std::vector<int32_t> routes[2]; // layers 10 and 11: [n_decode][3][10]
    std::vector<int32_t> cells;     // layer 11: [n_decode][3][2051]
    std::vector<int32_t> counts;    // layer 11: [n_decode][3]
};

// dir: the capture directory. False (err set) on a missing file or a malformed line.
bool load_hx7(const std::string & dir, hx7_traffic & t, std::string & err);
// The first n_check decodes read again straight from the files, line by line, without the loader's path; false (err)
// on any difference in an ID, a cell or a count (xp1-test's check of the loader).
bool check_hx7(const std::string & dir, const hx7_traffic & t, int32_t n_check, std::string & err);

} // namespace fn::xp1
