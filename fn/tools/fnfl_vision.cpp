// VIS1 Stage A: the vision module's tests. fnfl-vision-host (no CUDA; also the sanitizer build) runs the host
// commands; fnfl-vision-test runs those and the GPU commands (one card, the vision file only, no model load).
//
//   prep    REF_DIR IMAGE...          the normalized image bitwise against the reference's dumps; the patch cut
//   limits  IMAGES_DIR                bounds and bad-image cases, the size rule's branches
//   fuzz    SEED ITERS FILE...        mutated JPEG and PNG files through the bounded decode
//   range   MMPROJ                    the load-time FP16 range proof
//   semantics MMPROJ IMAGE PREFIX     each FP64 step on the reference's own layer-0 tensors (vis-ref --dump)
//   ref64   MMPROJ REF_DIR THREADS IMAGE...   the FP64 encoder against the reference engine's rows
//   ref16   MMPROJ REF_DIR THREADS IMAGE...   FP16 storage emulated in FP64, against FP64, the reference and the GPU
//   rows    SEED                      Stage B's image ledger (fn/runtime/image_rows): planes, offsets, keys, rules
//   kernels DEVICE                    each kernel against FP64 (seeded, three grids), a block and the merger
//   encoder MMPROJ REF_DIR DEVICE IMAGE...    the encoder against FP64 and the reference engine
//   cap     MMPROJ DEVICE IMAGE       the scratch at the 4,096-token cap; one encode's time, printed
//   resources DEVICE                  each kernel's registers, spills, shared memory and residency

#include "fn/runtime/image_rows_test.h"
#include "fn/vision/vision_test.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::vector<std::string> rest(int argc, char ** argv, int from) {
    std::vector<std::string> v;
    for (int i = from; i < argc; ++i) {
        v.emplace_back(argv[i]);
    }
    return v;
}

int usage() {
    std::fprintf(
        stderr,
        "usage: fnfl-vision-test prep REF_DIR IMAGE... | limits IMAGES_DIR | fuzz SEED ITERS FILE... | "
        "range MMPROJ | ref64 MMPROJ REF_DIR THREADS IMAGE... | ref16 MMPROJ REF_DIR THREADS IMAGE... | rows SEED%s\n",
#ifdef FNFL_VISION_HOST_ONLY
        ""
#else
        " | kernels DEVICE | encoder MMPROJ REF_DIR DEVICE IMAGE... | cap MMPROJ DEVICE IMAGE | "
        "resources DEVICE"
#endif
    );
    return 2;
}

}  // namespace

int main(int argc, char ** argv) {
    namespace t = fn::vision::test;
    if (argc < 3) {
        return usage();
    }
    const std::string cmd = argv[1];
    if (cmd == "prep" && argc >= 4) {
        return t::run_prep(rest(argc, argv, 3), argv[2]);
    }
    if (cmd == "limits") {
        return t::run_limits(argv[2]);
    }
    if (cmd == "fuzz" && argc >= 5) {
        return t::run_fuzz(rest(argc, argv, 4), std::atoi(argv[3]), uint32_t(std::strtoul(argv[2], nullptr, 10)));
    }
    if (cmd == "rows") {
        return fn::runtime::image_rows_test(uint32_t(std::strtoul(argv[2], nullptr, 10)));
    }
    if (cmd == "range") {
        return t::run_range(argv[2]);
    }
    if (cmd == "semantics" && argc >= 5) {
        return t::run_semantics(argv[2], argv[3], argv[4]);
    }
    if (cmd == "ref64" && argc >= 6) {
        return t::run_ref64(argv[2], rest(argc, argv, 5), argv[3], std::atoi(argv[4]));
    }
    if (cmd == "ref16" && argc >= 6) {
        return t::run_ref16(argv[2], rest(argc, argv, 5), argv[3], std::atoi(argv[4]));
    }
#ifndef FNFL_VISION_HOST_ONLY
    if (cmd == "kernels") {
        return t::run_kernels(std::atoi(argv[2]));
    }
    if (cmd == "encoder" && argc >= 6) {
        return t::run_encoder(argv[2], rest(argc, argv, 5), argv[3], std::atoi(argv[4]), 8, false);
    }
    if (cmd == "cap" && argc >= 5) {
        return t::run_cap(argv[2], argv[4], std::atoi(argv[3]));
    }
    if (cmd == "resources") {
        return t::run_resources(std::atoi(argv[2]));
    }
#endif
    return usage();
}
