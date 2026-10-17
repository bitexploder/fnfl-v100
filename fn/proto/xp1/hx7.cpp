#include "fn/proto/xp1/hx7.h"

#include <cstdio>
#include <cstring>
#include <memory>

namespace fn::xp1 {
namespace {

constexpr int32_t n_used = 10;

struct pipe_closer {
    void operator()(FILE * f) const {
        if (f != nullptr) {
            pclose(f);
        }
    }
};
using pipe_ptr = std::unique_ptr<FILE, pipe_closer>;

pipe_ptr gunzip(const std::string & path) {
    const std::string cmd = "gzip -dc '" + path + "'";
    return pipe_ptr(popen(cmd.c_str(), "r"));
}

bool fail(std::string & err, const std::string & what) {
    err = "hx7: " + what;
    return false;
}

} // namespace

bool load_hx7(const std::string & dir, hx7_traffic & t, std::string & err) {
    t = hx7_traffic();
    // the selection index: which file rows are layer 11's, and their decode and row
    FILE * fi = fopen((dir + "/routes.txt.sel.idx").c_str(), "r");
    if (fi == nullptr) {
        return fail(err, "cannot read routes.txt.sel.idx");
    }
    struct row_at {
        int32_t decode, b, n;
    };
    std::vector<row_at> idx; // per file row; b < 0 for other layers
    int32_t d = 0, row = 0, layer = 0, n = 0;
    while (fscanf(fi, "%d %d %d %d", &d, &row, &layer, &n) == 4) {
        idx.push_back({d, layer == 11 ? row - hx7_rows * d : -1, n});
        t.n_decode = d + 1 > t.n_decode ? d + 1 : t.n_decode;
    }
    fclose(fi);
    if (t.n_decode < 1 || idx.empty()) {
        return fail(err, "empty selection index");
    }
    const size_t nd = static_cast<size_t>(t.n_decode);
    t.cells.assign(nd * hx7_rows * hx7_width, -1);
    t.counts.assign(nd * hx7_rows, -1);
    t.routes[0].assign(nd * hx7_rows * n_used, -1);
    t.routes[1].assign(nd * hx7_rows * n_used, -1);
    {
        pipe_ptr fs = gunzip(dir + "/routes.txt.sel.gz");
        if (!fs) {
            return fail(err, "cannot open routes.txt.sel.gz");
        }
        std::vector<int32_t> buf(hx7_width);
        for (const row_at & r : idx) {
            if (fread(buf.data(), 4, hx7_width, fs.get()) != size_t(hx7_width)) {
                return fail(err, "routes.txt.sel.gz is shorter than its index");
            }
            if (r.b < 0) {
                continue;
            }
            if (r.b >= hx7_rows || r.decode < 0) {
                return fail(err, "bad selection row");
            }
            const size_t at = static_cast<size_t>(r.decode) * hx7_rows + static_cast<size_t>(r.b);
            std::memcpy(&t.cells[at * hx7_width], buf.data(), size_t(hx7_width) * 4);
            t.counts[at] = r.n;
        }
    }
    {
        pipe_ptr fr = gunzip(dir + "/routes.txt.gz");
        if (!fr) {
            return fail(err, "cannot open routes.txt.gz");
        }
        while (fscanf(fr.get(), "%d %d %d", &d, &row, &layer) == 3) {
            int32_t e[n_used];
            for (int32_t k = 0; k < n_used; ++k) {
                if (fscanf(fr.get(), "%d", &e[k]) != 1 || e[k] < 0 || e[k] >= 512) {
                    return fail(err, "bad expert ID in routes.txt.gz");
                }
            }
            if (layer != 10 && layer != 11) {
                continue;
            }
            const int32_t b = row - hx7_rows * d;
            if (d < 0 || d >= t.n_decode || b < 0 || b >= hx7_rows) {
                return fail(err, "bad route row");
            }
            std::memcpy(&t.routes[layer - 10][(static_cast<size_t>(d) * hx7_rows + b) * n_used], e, sizeof(e));
        }
    }
    for (const int32_t v : t.counts) {
        if (v < 0) {
            return fail(err, "a decode row has no layer-11 selection");
        }
    }
    for (int32_t l = 0; l < 2; ++l) {
        for (const int32_t v : t.routes[l]) {
            if (v < 0) {
                return fail(err, "a decode row has no layer-" + std::to_string(10 + l) + " route");
            }
        }
    }
    return true;
}

bool check_hx7(const std::string & dir, const hx7_traffic & t, int32_t n_check, std::string & err) {
    // IDs: the route lines of the first n_check decodes
    pipe_ptr fr = gunzip(dir + "/routes.txt.gz");
    if (!fr) {
        return fail(err, "cannot open routes.txt.gz");
    }
    char line[512];
    int32_t seen = 0;
    while (fgets(line, sizeof(line), fr.get()) != nullptr) {
        int32_t d = 0, row = 0, layer = 0, e[n_used];
        if (sscanf(line, "%d %d %d %d %d %d %d %d %d %d %d %d %d", &d, &row, &layer, &e[0], &e[1], &e[2], &e[3], &e[4],
                   &e[5], &e[6], &e[7], &e[8], &e[9]) != 13) {
            return fail(err, std::string("unreadable route line: ") + line);
        }
        if (d >= n_check) {
            break;
        }
        if (layer == 10 || layer == 11) {
            const size_t at = (static_cast<size_t>(d) * hx7_rows + (row - hx7_rows * d)) * n_used;
            for (int32_t k = 0; k < n_used; ++k) {
                if (t.routes[layer - 10][at + k] != e[k]) {
                    return fail(err, "route of decode " + std::to_string(d) + " differs");
                }
            }
            ++seen;
        }
    }
    if (seen != n_check * hx7_rows * 2) {
        return fail(err, "route lines checked: " + std::to_string(seen));
    }
    // cells: the counts from the index, and each row's cells below its count from the gzip, in file order
    FILE * fi = fopen((dir + "/routes.txt.sel.idx").c_str(), "r");
    pipe_ptr fs = gunzip(dir + "/routes.txt.sel.gz");
    if (fi == nullptr || !fs) {
        if (fi != nullptr) {
            fclose(fi);
        }
        return fail(err, "cannot open the selection files");
    }
    std::vector<int32_t> buf(hx7_width);
    int32_t d = 0, row = 0, layer = 0, n = 0, checked = 0;
    bool ok = true;
    while (ok && fscanf(fi, "%d %d %d %d", &d, &row, &layer, &n) == 4 && d < n_check) {
        ok = fread(buf.data(), 4, hx7_width, fs.get()) == size_t(hx7_width);
        if (ok && layer == 11) {
            const size_t at = static_cast<size_t>(d) * hx7_rows + (row - hx7_rows * d);
            ok = t.counts[at] == n && n >= 2048 && n <= hx7_width;
            for (int32_t i = 0; ok && i < hx7_width; ++i) {
                ok = t.cells[at * hx7_width + i] == buf[i] && buf[i] >= 0 && buf[i] <= 58583;
            }
            ++checked;
        }
    }
    fclose(fi);
    if (!ok || checked != n_check * hx7_rows) {
        return fail(err, "selection of the first decodes differs (checked " + std::to_string(checked) + ")");
    }
    return true;
}

} // namespace fn::xp1
