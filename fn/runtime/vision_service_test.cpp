#include "fn/runtime/vision_service_test.h"

#include "fn/loader/run_load.h"
#include "fn/model/model.h"
#include "fn/model/state.h"
#include "fn/runtime/image_rows.h"
#include "fn/runtime/vision_service.h"
#include "fn/vision/encoder.h"
#include "fn/vision/image.h"
#include "fn/vision/sizes.h"
#include "fn/vision/weights.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace fn::runtime {

namespace {

constexpr int32_t W = vision::out_width, H = W / 2;

int fails = 0;

void check(bool ok, const std::string & what) {
    std::printf("vis-test: %s: %s\n", what.c_str(), ok ? "PASS" : "FAIL");
    fails += ok ? 0 : 1;
}

bool read_file(const char * path, std::vector<uint8_t> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return !out.empty();
}

// card k's supplied rows of a chunk (its runs' total) against the reference rows k0 .. k0 + n - 1, half k
bool supplied_ok(const model::prompt_chunk & c,
                 int32_t                     k,
                 int32_t                     dev,
                 const std::vector<float> &  ref,
                 int64_t                     k0,
                 int32_t                     n) {
    std::vector<float> got(size_t(n) * H);
    if (c.embd[k] == nullptr || cudaSetDevice(dev) != cudaSuccess ||
        cudaMemcpy(got.data(), c.embd[k], got.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    for (int32_t r = 0; r < n; ++r) {
        if (std::memcmp(&got[size_t(r) * H], &ref[size_t(k0 + r) * W + size_t(k) * H], size_t(H) * 4) != 0) {
            std::printf("vis-test: card %d's supplied row %d (image row %lld) differs\n", k, r,
                        static_cast<long long>(k0 + r));
            return false;
        }
    }
    return true;
}

bool planes_ok(const model::prompt_chunk & c, int32_t k, int32_t dev, const image_chunk & h) {
    std::vector<int32_t> p(h.planes.size()), b(h.bplanes.size());
    if (c.planes[k] == nullptr || c.bplanes[k] == nullptr || cudaSetDevice(dev) != cudaSuccess ||
        cudaMemcpy(p.data(), c.planes[k], p.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
        (!b.empty() && cudaMemcpy(b.data(), c.bplanes[k], b.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess)) {
        return false;
    }
    return p == h.planes && b == h.bplanes;
}

}  // namespace

int vision_service_test(int32_t dev0, int32_t dev1, const char * mmproj, const char * image) {
    fails = 0;
    std::string          err;
    std::vector<uint8_t> bytes;
    auto                 img = std::make_shared<vision::prepared>();
    if (!read_file(image, bytes) ||
        vision::prepare(bytes.data(), bytes.size(), vision::limits(), *img) != vision::status::ok) {
        std::printf("vis-test: cannot prepare %s\n", image);
        return 2;
    }
    const int32_t T = img->nx * img->ny;
    std::printf("vis-test: %s: %d x %d tokens (%d), devices %d and %d (no text model)\n", image, img->nx, img->ny, T,
                dev0, dev1);

    // the reference rows: one card's whole encoder (Stage A's path) on dev1
    std::vector<float> ref(size_t(T) * W);
    {
        vision::host_weights   hw;
        vision::device_weights w;
        vision::encoder        e;
        float *                out = nullptr;
        const bool ok = vision::load_host(mmproj, hw, err) && w.upload(hw, dev1, 0, vision::blocks, true, true, err) &&
                        e.init(dev1, vision::max_patches, err) && cudaSetDevice(dev1) == cudaSuccess &&
                        cudaMalloc(&out, ref.size() * 4) == cudaSuccess && e.encode(w, *img, out) && e.sync() &&
                        cudaMemcpy(ref.data(), out, ref.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
        if (out != nullptr) {
            cudaFree(out);
        }
        if (!ok) {
            std::printf("vis-test: the reference encode: %s\n", err.c_str());
            return 2;
        }
    }

    // the service on its classes alone (two lanes)
    model::model m;
    m.device[0] = dev0;
    m.device[1] = dev1;
    m.single    = dev0 == dev1;
    model::vision_params vp;
    vp.weights[0]             = int64_t(vision::weight_bytes(0, vision::vision_split, true, false));
    vp.weights[1]             = int64_t(vision::weight_bytes(vision::vision_split, vision::blocks, false, true));
    vp.scratch                = int64_t(vision::plan_scratch(vision::max_patches).total);
    vp.patch_stage            = int64_t(vision::patch_stage_bytes(vision::max_patches));
    vp.lanes                  = 2;
    int64_t        host_bytes = 0;
    vision_service vs;
    if (!loader::reserve_vision(m, vp, false, host_bytes, err) || !vs.init(m, mmproj, err)) {
        std::printf("vis-test: the service: %s\n", err.c_str());
        return 2;
    }
    void * streams[2] = { nullptr, nullptr };
    for (int32_t k = 0; k < 2; ++k) {
        cudaStream_t s = nullptr;
        if (cudaSetDevice(m.device[k]) != cudaSuccess ||
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
            std::printf("vis-test: streams\n");
            return 2;
        }
        streams[k] = s;
    }
    const auto sync = [&] {
        bool ok = true;
        for (int32_t k = 0; k < 2; ++k) {
            ok = cudaSetDevice(m.device[k]) == cudaSuccess &&
                 cudaStreamSynchronize(static_cast<cudaStream_t>(streams[k])) == cudaSuccess && ok;
        }
        return ok && cudaGetLastError() == cudaSuccess;
    };

    // the prompt: 5 text tokens, the image, 8 text tokens (the end rule); two chunks cutting the image in the middle
    const int64_t        c0 = 5, n = c0 + T + 8;
    std::vector<int32_t> tok(size_t(n), 17);
    std::fill(tok.begin() + c0, tok.begin() + c0 + T, image_token);
    image_rows l;
    if (!l.set(tok.data(), n,
               {
                   { c0, img->nx, img->ny, img->digest }
    },
               err) ||
        !l.end_ok(n, err)) {
        std::printf("vis-test: the ledger: %s\n", err.c_str());
        return 2;
    }
    const vision_service::pixels px = [&](size_t i) {
        return i == 0 ? img : nullptr;
    };
    const int64_t cut = c0 + T / 2;
    const int32_t ra = int32_t(cut), rb = int32_t(n - cut);
    image_chunk   ha, hb;
    l.chunk(0, ra, ha);
    l.chunk(cut, rb, hb);

    model::prompt_chunk a;
    a.start = 0;
    a.rows  = ra;
    bool ok = vs.fill_rows(0, l, px, streams, 0, ha, a, false, err) && sync();
    check(ok && vs.encodes() == 1 && a.n_runs == 1 && a.runs[0].row0 == c0 && a.runs[0].n == T / 2 &&
              supplied_ok(a, 0, dev0, ref, 0, T / 2) && supplied_ok(a, 1, dev1, ref, 0, T / 2),
          "chunk 1 (the image's first half): one encode, both cards' supplied halves bitwise the one-card encoder's" +
              (ok ? std::string() : ": " + err));
    check(ok && planes_ok(a, 0, dev0, ha) && planes_ok(a, 1, dev1, ha), "chunk 1's rope and block planes, both cards");

    model::prompt_chunk b;
    b.start = cut;
    b.rows  = rb;
    ok      = vs.fill_rows(0, l, px, streams, 0, hb, b, false, err) && sync();
    check(ok && vs.encodes() == 1 && b.n_runs == 1 && b.runs[0].row0 == 0 && b.runs[0].n == T - T / 2 &&
              supplied_ok(b, 0, dev0, ref, T / 2, T - T / 2) && supplied_ok(b, 1, dev1, ref, T / 2, T - T / 2),
          "chunk 2 (the second half): the held image copied, no second encode" + (ok ? std::string() : ": " + err));
    check(ok && planes_ok(b, 0, dev0, hb) && planes_ok(b, 1, dev1, hb), "chunk 2's rope and block planes, both cards");

    model::prompt_chunk b1;
    b1.start = cut;
    b1.rows  = rb;
    ok       = vs.fill_rows(0, l, px, streams, 1, hb, b1, false, err) && sync();
    check(ok && vs.encodes() == 2 && supplied_ok(b1, 0, dev0, ref, T / 2, T - T / 2) &&
              supplied_ok(b1, 1, dev1, ref, T / 2, T - T / 2),
          "chunk 2 in the other lane: encoded there once, the same rows" + (ok ? std::string() : ": " + err));

    // refusals: another slot (nothing held for it) without its prepared image; a digest that differs from the span
    model::prompt_chunk r;
    r.start = 0;
    r.rows  = ra;
    const bool no_px =
        !vs.fill_rows(
            1, l, [](size_t) { return std::shared_ptr<const vision::prepared>(); }, streams, 0, ha, r, false, err) &&
        err.find("no prepared image") != std::string::npos && !vs.pending(0);
    auto other = std::make_shared<vision::prepared>(*img);
    other->digest ^= 1;
    const bool differs = !vs.fill_rows(
                             2, l, [&](size_t) { return other; }, streams, 0, ha, r, false, err) &&
                         err.find("differs from its span") != std::string::npos && !vs.pending(0);
    check(no_px && differs && sync(), "refusals (no prepared image; a digest other than the span's)");

    // slices (the job steps): chunk 1 for another slot, one encoder block a step: 27 steps, the same rows
    model::prompt_chunk e1;
    e1.start        = 0;
    e1.rows         = ra;
    ok              = vs.fill_rows(3, l, px, streams, 0, ha, e1, true, err);
    int32_t n_steps = 0;
    while (ok && vs.pending(0) && n_steps < 64) {
        ok = vs.encode_step(0, 1, err);
        ++n_steps;
    }
    ok = ok && sync();
    check(ok && n_steps == vision::blocks && vs.encodes() == 3 && supplied_ok(e1, 0, dev0, ref, 0, T / 2) &&
              supplied_ok(e1, 1, dev1, ref, 0, T / 2) && planes_ok(e1, 0, dev0, ha),
          "sliced: " + std::to_string(n_steps) + " one-block steps, the same rows bitwise" +
              (ok ? std::string() : ": " + err));

    // an abandoned encode: five slices, forget, then the chunk again in one call (the partial encode discarded)
    model::prompt_chunk f1;
    f1.start = 0;
    f1.rows  = ra;
    ok       = vs.fill_rows(4, l, px, streams, 0, ha, f1, true, err);
    for (int32_t i = 0; i < 5 && ok; ++i) {
        ok = vs.encode_step(0, 1, err);
    }
    const bool mid = ok && vs.pending(0) && vs.encodes() == 3;
    vs.forget(0);
    ok = mid && !vs.pending(0) && vs.fill_rows(4, l, px, streams, 0, ha, f1, false, err) && sync();
    check(ok && vs.encodes() == 4 && supplied_ok(f1, 0, dev0, ref, 0, T / 2) && supplied_ok(f1, 1, dev1, ref, 0, T / 2),
          "forget mid-encode, then the chunk again: the same rows" + (ok ? std::string() : ": " + err));

    // two lanes queued: lane 1's step finishes lane 0's image first (one encoder, queue order), then its own
    model::prompt_chunk g0, g1;
    g0.start = g1.start = 0;
    g0.rows = g1.rows = ra;
    ok = vs.fill_rows(5, l, px, streams, 0, ha, g0, true, err) && vs.fill_rows(6, l, px, streams, 1, ha, g1, true, err);
    ok = ok && vs.encode_step(1, vision::blocks, err);
    const bool order = ok && !vs.pending(0) && vs.pending(1) && vs.encodes() == 5;
    ok               = order && vs.encode_step(1, vision::blocks, err) && !vs.pending(1) && sync();
    check(ok && vs.encodes() == 6 && supplied_ok(g0, 0, dev0, ref, 0, T / 2) && supplied_ok(g1, 1, dev1, ref, 0, T / 2),
          "two lanes in queue order, each its own rows" + (ok ? std::string() : ": " + err));

    for (int32_t k = 0; k < 2; ++k) {
        cudaSetDevice(m.device[k]);
        cudaStreamDestroy(static_cast<cudaStream_t>(streams[k]));
    }
    std::printf("vis-test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

}  // namespace fn::runtime
