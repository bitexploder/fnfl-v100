#pragma once

// VIS1 Stage A tests. Host (no card): preprocessing bitwise against the reference's dumps, the decoders' bounds and
// bad-image cases, mutated files under the sanitizers, the load-time range proof, and the FP64 encoder against the
// reference engine's rows. GPU (one card, the vision file only, no model load): each kernel against FP64 over seeded
// inputs at three grids, the whole encoder against FP64 and the reference engine, and the scratch at the cap.
//
// Declared tolerances (chosen from the arithmetic before the runs; kernels.h has each kernel's): per row,
// max |gpu - ref| / max |ref| at most
//   pos_resize 2e-6, patch_embed 1e-5, layernorm / qkv_rope / bias_gelu 1e-3, attention 4e-3, one synthetic block
//   (the products and chunk loops composed) 5e-3, merger chain 5e-3; bias_residual and bias_add bitwise.
// The whole encoder against FP64: per output row, ||gpu - ref||_2 / ||ref||_2 at most 2e-2 (max over rows) and at
// most 5e-3 (mean over rows): FP16's unit roundoff is 4.9e-4, about eight roundings a block over 27 blocks and the
// merger give about 7e-3 rms if they add as a random walk, with room for the network's amplification. The reference
// engine comparison is reported, not judged (R16).

#include <cstdint>
#include <string>
#include <vector>

namespace fn::vision::test {

// tol_pos: F32 grid coordinates (ggml's sx = gx / ((gw - 1) / 47), two roundings) are off by at most 47 x 2^-23,
// 5.6e-6, in each axis; with taps up to 2 max|table| apart that is 2.2e-5 x max|table| over both axes, plus the
// bilinear's own rounding (about 8 ulp, 5e-7). The tests' rows reach max|table| within a few percent, so 4e-5 of the
// row's max bounds it.
constexpr double tol_pos = 4e-5, tol_patch = 1e-5, tol_fp16 = 1e-3, tol_attn = 4e-3, tol_merger = 5e-3,
                 tol_block      = 5e-3;
// the whole encoder (engineer's ruling, VIS1 2026-10-27; the fixed 2e-2 / 5e-3 bounds are withdrawn): per image with
// FP64 rows, the GPU's mean per-row error against FP64 at most tol_enc_mean_x times the FP16-storage reference's
// (ref16), and its worst row at most the reference engine's own worst row against FP64
constexpr double tol_enc_mean_x = 1.5;

// a VIS1DMP dump (work/VIS1/ref/vis_ref.cpp's format)
struct dump {
    int32_t            type = 0, ndim = 0;
    int64_t            dims[4] = { 1, 1, 1, 1 };
    std::vector<float> f32;
};

bool read_dump(const std::string & path, dump & d);

struct row_err {
    double  max_rel = 0, mean_rel = 0, max_abs_over_max = 0;
    int32_t worst_row = -1;
};

// per-row relative L2 errors of a against ref ([rows][n])
row_err compare_rows(const float * a, const double * ref, int32_t rows, int32_t n);
row_err compare_rows(const float * a, const float * ref, int32_t rows, int32_t n);

// host commands (fnfl-vision-host and fnfl-vision): each returns 0 for PASS
int run_prep(const std::vector<std::string> & images, const std::string & ref_dir);
int run_limits(const std::string & images_dir);  // test-640x488.jpg and photo-224.png from it
int run_fuzz(const std::vector<std::string> & files, int32_t iters, uint32_t seed);
int run_range(const std::string & mmproj);
// each step of the FP64 reference applied to the reference engine's own layer-0 inputs (vis-ref --dump), against its
// outputs: the semantics, step by step, without the accumulated noise
int run_semantics(const std::string & mmproj, const std::string & image, const std::string & prefix);
int run_ref64(const std::string &              mmproj,
              const std::vector<std::string> & images,
              const std::string &              ref_dir,
              int32_t                          threads);

// the FP64 encoder with FP16 storage emulated (ref::set_fp16_storage), against the FP64 rows, the reference engine's
// rows and the GPU's rows (REF_DIR/<image>.gpu.bin, written by the encoder command) when present
int run_ref16(const std::string &              mmproj,
              const std::vector<std::string> & images,
              const std::string &              ref_dir,
              int32_t                          threads);

// GPU commands (fnfl-vision)
int run_kernels(int32_t device);
int run_encoder(const std::string &              mmproj,
                const std::vector<std::string> & images,
                const std::string &              ref_dir,
                int32_t                          device,
                int32_t                          threads,
                bool                             fp64);
int run_cap(const std::string & mmproj, const std::string & image, int32_t device);
int run_resources(int32_t device);

}  // namespace fn::vision::test
