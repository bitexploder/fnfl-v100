// ME2: the CPU miss path (experts_cpu.h): pack_gate_up / pack_down's inverses, the activations, and Strata-V100
// 3d4ce4e0's native-expert pool (src/kernels/cpu/pool.cpp: publish, worker, claim, begin_batch, wait_parked, wait_done,
// run_phase and the mode 5 / 6 arms of drain; include/strata/kernels/cpu/pool.hpp). Strata's notice:
//
// MIT License
//
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Contract (the inverses): unpack_gate_up_tile and unpack_down_rows write every byte of their native rows from the
// packed bytes pack_gate_up / pack_down (experts.cu, 727735b) placed, so pack then unpack returns the GGUF bytes
// (ex-cpu-test checks it, every format). The K-quants' 12 scale bytes are rebuilt bit field by bit field from the
// packed 6-bit scales and mins (get_scale_min_k4's inverse).

#include "fn/kernels/experts/experts_cpu.h"
#include "fn/quant/formats.h"

#include <pthread.h>
#include <sched.h>

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace fn::experts::cpu {

namespace {

// ---- the packed gate/up layout (experts_dev.cuh: gu_layout_k, tile_bytes, aux8_off, sc8_off, d_off, sm_off,
// dmin_off at 727735b) ----
constexpr int32_t gu_threads = 640; // records a tile: rec = 8 j + rs
constexpr int32_t n_sub = d_model / 32;

struct layout {
    int32_t block_bytes, chunks;
    bool aux8, sc8;
};

constexpr layout layout_of(fmt f) {
    return f == fmt::iq2_xxs   ? layout{66, 4, false, false}
           : f == fmt::iq2_xs  ? layout{74, 4, true, false}
           : f == fmt::iq2_s   ? layout{82, 5, false, false}
           : f == fmt::iq3_xxs ? layout{98, 6, false, false}
           : f == fmt::iq3_s   ? layout{110, 6, true, true}
           : f == fmt::q4_k    ? layout{148, 8, false, false}
           : f == fmt::q5_k    ? layout{180, 10, false, false}
                               : layout{0, 0, false, false};
}
constexpr bool is_kq(fmt f) {
    return f == fmt::q4_k || f == fmt::q5_k;
}
constexpr int64_t tile_bytes(fmt f) {
    return int64_t(gu_threads) * layout_of(f).block_bytes;
}
constexpr int64_t aux8_off(fmt f) {
    return int64_t(layout_of(f).chunks) * gu_threads * 16;
}
constexpr int64_t sc8_off(fmt f) {
    return aux8_off(f) + (layout_of(f).aux8 ? gu_threads * 8 : 0);
}
constexpr int64_t d_off(fmt f) {
    return sc8_off(f) + (layout_of(f).sc8 ? (n_sub / 2) * 8 * 8 : 0) + (is_kq(f) ? gu_threads * 16 : 0);
}
constexpr int64_t sm_off(fmt f) {
    return aux8_off(f);
}
constexpr int64_t dmin_off(fmt f) {
    return d_off(f) + 10 * 8 * 16;
}
static_assert(d_off(fmt::iq3_s) + 10 * 8 * 16 == tile_bytes(fmt::iq3_s), "IQ3_S tile");
static_assert(d_off(fmt::iq2_s) + 10 * 8 * 16 == tile_bytes(fmt::iq2_s), "IQ2_S tile");
static_assert(dmin_off(fmt::q4_k) + 10 * 8 * 16 == tile_bytes(fmt::q4_k), "Q4_K tile");
static_assert(dmin_off(fmt::q5_k) + 10 * 8 * 16 == tile_bytes(fmt::q5_k), "Q5_K tile");

// GGUF bytes a 256-weight block (the K-quants' are 4 fewer than their packed blocks: the scales stay packed)
constexpr int32_t gguf_block(fmt f) {
    return f == fmt::q4_k ? 144 : f == fmt::q5_k ? 176 : layout_of(f).block_bytes;
}

// ---- the packed down layout (experts.cu pack_down at 727735b): D output rows, 5 items of 64 columns a card ----
constexpr int32_t n_item_k = 5;

inline void cp(uint8_t * dst, const uint8_t * src, size_t n) {
    std::memcpy(dst, src, n);
}

// tile -> 64 native rows; the GGUF block of record (j, row r) is dst + r rb + (j >> 3) bb
template <fmt F> void unpack_tile(const uint8_t * tile, uint8_t * dst) {
    constexpr int32_t bb = gguf_block(F);
    constexpr int64_t rb = int64_t(10) * bb;
    if constexpr (is_kq(F)) {
        std::memset(dst, 0, size_t(64 * rb)); // the scale bytes and nibbles are set field by field
    }
    for (int32_t rec = 0; rec < gu_threads; ++rec) {
        const int32_t j = rec >> 3, rs = rec & 7;
        const int32_t sbk = j >> 3, ib = j & 7;
        const uint8_t * c0 = tile + int64_t(rec) * 16; // chunk c of the record: c0 + c x 640 x 16
        const auto mc = [&](int32_t c, int32_t byte) { return c0 + int64_t(c) * gu_threads * 16 + byte; };
        for (int32_t mrow = 0; mrow < 8; ++mrow) {
            const int32_t r = rs + 8 * mrow;
            uint8_t * blk = dst + r * rb + sbk * bb;
            if constexpr (F == fmt::iq2_xxs || F == fmt::iq2_xs) {
                cp(blk + 2 + 8 * ib, mc(mrow / 2, 8 * (mrow & 1)), 8);
                if constexpr (F == fmt::iq2_xs) {
                    blk[2 + 64 + ib] = tile[aux8_off(F) + rec * 8 + mrow];
                }
            } else if constexpr (F == fmt::iq2_s) {
                blk[2 + 64 + ib] = *mc(0, mrow);                               // qh
                blk[2 + 64 + 8 + ib] = *mc(0, 8 + mrow);                       // scales
                cp(blk + 2 + 4 * ib, mc(1 + mrow / 2, 8 * (mrow & 1)), 4);      // grid low bits
                cp(blk + 2 + 32 + 4 * ib, mc(1 + mrow / 2, 8 * (mrow & 1) + 4), 4); // signs
            } else if constexpr (F == fmt::iq3_xxs || F == fmt::iq3_s) {
                const int32_t hq = mrow / 4, i4 = mrow & 3;
                constexpr int32_t q_chunk[4] = {0, 1, 2, 2}, q_byte[4] = {0, 0, 0, 8};
                constexpr int32_t s_chunk[4] = {0, 0, 1, 1}, s_byte[4] = {8, 12, 8, 12};
                cp(blk + 2 + 8 * ib, mc(3 * hq + q_chunk[i4], q_byte[i4]), 8);
                if constexpr (F == fmt::iq3_xxs) {
                    cp(blk + 2 + 64 + 4 * ib, mc(3 * hq + s_chunk[i4], s_byte[i4]), 4);
                } else {
                    cp(blk + 2 + 64 + 8 + 4 * ib, mc(3 * hq + s_chunk[i4], s_byte[i4]), 4); // signs
                    blk[2 + 64 + ib] = tile[aux8_off(F) + rec * 8 + mrow];                  // qh
                    if ((j & 1) == 0) {
                        blk[2 + 64 + 8 + 32 + ib / 2] = tile[sc8_off(F) + ((j >> 1) * 8 + rs) * 8 + mrow];
                    }
                }
            } else { // Q4_K, Q5_K
                constexpr bool q5 = F == fmt::q5_k;
                const int32_t chunk = q5 ? (mrow < 4 ? 1 + mrow : 2 + mrow) : mrow;
                const uint8_t * pk = mc(chunk, 0);
                uint8_t * qs = blk + (q5 ? 48 : 16) + 32 * (ib / 2);
                const int32_t sh = (ib & 1) ? 4 : 0;
                for (int32_t t = 0; t < 16; ++t) {
                    qs[t] = static_cast<uint8_t>(qs[t] | ((pk[t] & 0xf) << sh));
                    qs[16 + t] = static_cast<uint8_t>(qs[16 + t] | ((pk[t] >> 4) << sh));
                }
                if constexpr (q5) {
                    uint32_t hw;
                    std::memcpy(&hw, mc(mrow < 4 ? 0 : 5, 4 * (mrow & 3)), 4);
                    uint8_t * qh = blk + 16;
                    for (int32_t t = 0; t < 32; ++t) {
                        qh[t] = static_cast<uint8_t>(qh[t] | (((hw >> t) & 1u) << ib));
                    }
                }
                const uint8_t * sm = tile + sm_off(F) + int64_t(rec) * 16 + 2 * mrow;
                const uint32_t sc = sm[0], mn = sm[1]; // 6 bits each
                uint8_t * q = blk + 4;
                if (ib < 4) {
                    q[ib] = static_cast<uint8_t>(q[ib] | sc);
                    q[ib + 4] = static_cast<uint8_t>(q[ib + 4] | mn);
                } else {
                    q[ib + 4] = static_cast<uint8_t>((sc & 0xf) | ((mn & 0xf) << 4));
                    q[ib - 4] = static_cast<uint8_t>(q[ib - 4] | ((sc >> 4) << 6));
                    q[ib] = static_cast<uint8_t>(q[ib] | ((mn >> 4) << 6));
                }
                if (ib == 0) {
                    cp(blk + 2, tile + dmin_off(F) + (sbk * 8 + rs) * 16 + 2 * mrow, 2);
                }
            }
            if (ib == 0) {
                cp(blk, tile + d_off(F) + (sbk * 8 + rs) * 16 + 2 * mrow, 2); // d
            }
        }
    }
}

// one card's 320 columns of output rows n0 .. n0 + n - 1 into dst (row i at i x rb + col_off)
void unpack_down_card(fmt f, const uint8_t * w, int32_t n0, int32_t n, uint8_t * dst, int64_t rb, int64_t col_off) {
    if (f == fmt::q5_1) { // 10 blocks of 24 bytes (FP16 d, FP16 m, qh word, 16 qs bytes) a card
        const int64_t d_plane = int64_t(n_item_k) * 2 * d_model * 16;
        for (int32_t b = 0; b < 2 * n_item_k; ++b) {
            for (int32_t i = 0; i < n; ++i) {
                const int32_t nn = n0 + i;
                uint8_t * blk = dst + i * rb + col_off + b * 24;
                const uint8_t * de = w + d_plane + (int64_t(b / 2) * d_model + nn) * 16;
                cp(blk + 8, w + (int64_t(b) * d_model + nn) * 16, 16);
                cp(blk + 4, de + 4 * (b & 1), 4);
                cp(blk, de + 8 + 4 * (b & 1), 4);
            }
        }
        return;
    }
    if (f == fmt::q8_0) { // 10 blocks of 34 bytes (FP16 d, 32 int8) a card
        constexpr int32_t nc8 = 4;
        const int64_t d_plane8 = int64_t(n_item_k) * nc8 * d_model * 16;
        for (int32_t b = 0; b < 2 * n_item_k; ++b) {
            const int32_t j5 = b / 2;
            for (int32_t i = 0; i < n; ++i) {
                const int32_t nn = n0 + i;
                uint8_t * blk = dst + i * rb + col_off + b * 34;
                for (int32_t h = 0; h < 2; ++h) {
                    cp(blk + 2 + 16 * h, w + ((int64_t(j5) * nc8 + 2 * (b & 1) + h) * d_model + nn) * 16, 16);
                }
                cp(blk, w + d_plane8 + (int64_t(j5) * d_model + nn) * 4 + 2 * (b & 1), 2);
            }
        }
        return;
    }
    // Q2_0 (5 blocks of 64 values) and IQ4_NL (10 of 32), 18 bytes each: FP16 d, 16 code bytes
    const int32_t nc = f == fmt::q2_0 ? 1 : 2;
    const int64_t d_plane = int64_t(n_item_k) * nc * d_model * 16;
    for (int32_t b = 0; b < n_item_k * nc; ++b) {
        for (int32_t i = 0; i < n; ++i) {
            const int32_t nn = n0 + i;
            uint8_t * blk = dst + i * rb + col_off + b * 18;
            cp(blk + 2, w + (int64_t(b) * d_model + nn) * 16, 16);
            cp(blk, w + d_plane + (f == fmt::q2_0 ? (int64_t(b) * d_model + nn) * 2
                                                  : (int64_t(b / 2) * d_model + nn) * 4 + 2 * (b & 1)),
               2);
        }
    }
}

} // namespace

bool gate_up_supported(fmt f) {
    return layout_of(f).block_bytes > 0 && gate_up_bytes(f) == 10 * tile_bytes(f);
}

bool down_supported(fmt f) {
    return f == fmt::q2_0 || f == fmt::iq4_nl || f == fmt::q5_1 || f == fmt::q8_0;
}

int64_t gu_row_bytes(fmt f) {
    return gate_up_supported(f) ? int64_t(10) * gguf_block(f) : 0;
}

int64_t dn_row_bytes(fmt f) {
    return f == fmt::q2_0 ? 180 : f == fmt::iq4_nl ? 360 : f == fmt::q5_1 ? 480 : f == fmt::q8_0 ? 680 : 0;
}

int64_t gu_scratch_bytes(fmt f) {
    return 64 * gu_row_bytes(f);
}

int64_t dn_scratch_bytes(fmt f) {
    return down_piece * dn_row_bytes(f);
}

void unpack_gate_up_tile(fmt f, const uint8_t * w, int32_t g, uint8_t * dst) {
    const uint8_t * tile = w + int64_t(g) * tile_bytes(f);
    switch (f) {
        case fmt::iq2_xxs:
            return unpack_tile<fmt::iq2_xxs>(tile, dst);
        case fmt::iq2_xs:
            return unpack_tile<fmt::iq2_xs>(tile, dst);
        case fmt::iq2_s:
            return unpack_tile<fmt::iq2_s>(tile, dst);
        case fmt::iq3_xxs:
            return unpack_tile<fmt::iq3_xxs>(tile, dst);
        case fmt::iq3_s:
            return unpack_tile<fmt::iq3_s>(tile, dst);
        case fmt::q4_k:
            return unpack_tile<fmt::q4_k>(tile, dst);
        case fmt::q5_k:
            return unpack_tile<fmt::q5_k>(tile, dst);
        default:
            return;
    }
}

void unpack_down_rows(fmt f, const uint8_t * w0, const uint8_t * w1, int32_t n0, int32_t n, uint8_t * dst) {
    const int64_t rb = dn_row_bytes(f);
    unpack_down_card(f, w0, n0, n, dst, rb, 0);
    unpack_down_card(f, w1, n0, n, dst, rb, rb / 2);
}

bool native_equal(fmt gu, fmt dn, const job & j, uint8_t * scratch, std::string & where) {
    const int64_t grb = gu_row_bytes(gu), drb = dn_row_bytes(dn);
    if (j.ngate == nullptr || j.nup == nullptr || j.ndown == nullptr || j.ngu_row != grb || j.ndn_row != drb) {
        where = "no native rows, or a row stride other than the format's";
        return false;
    }
    for (int32_t card = 0; card < 2; ++card) {
        for (int32_t g = 0; g < n_ch / 32; ++g) {
            unpack_gate_up_tile(gu, j.gu[card], g, scratch);
            const int64_t r0 = int64_t(card * n_ch + 32 * g) * grb;
            if (std::memcmp(scratch, j.ngate + r0, size_t(32 * grb)) != 0 ||
                std::memcmp(scratch + 32 * grb, j.nup + r0, size_t(32 * grb)) != 0) {
                where = "gate/up card " + std::to_string(card) + " tile " + std::to_string(g);
                return false;
            }
        }
    }
    for (int32_t n0 = 0; n0 < d_model; n0 += down_piece) {
        unpack_down_rows(dn, j.dn[0], j.dn[1], n0, down_piece, scratch);
        if (std::memcmp(scratch, j.ndown + int64_t(n0) * drb, size_t(down_piece * drb)) != 0) {
            where = "down rows " + std::to_string(n0) + " .. " + std::to_string(n0 + down_piece - 1);
            return false;
        }
    }
    return true;
}

void gate_up_act(const int8_t * codes, const float * scales, block_q8_k * act) {
    alignas(32) float x[d_model];
    for (int32_t k = 0; k < d_model; ++k) {
        x[k] = scales[2 * (k / 32)] * static_cast<float>(codes[k]);
    }
    quantize_q8_k(x, act, d_model);
}

void gate_up_act_f16(const uint16_t * x16, block_q8_k * act) {
    alignas(32) float x[d_model];
    for (int32_t k = 0; k < d_model; k += 8) {
        _mm256_store_ps(x + k, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x16 + k))));
    }
    quantize_q8_k(x, act, d_model);
}

bool to_f16(const float * x, int64_t n, uint16_t * out) {
    const __m256 lim = _mm256_set1_ps(65504.0f), abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 bad = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        // !(|v| <= 65,504): true for a larger value and for NaN (an unordered compare)
        bad = _mm256_or_ps(bad, _mm256_cmp_ps(_mm256_and_ps(v, abs), lim, _CMP_NLE_UQ));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
    }
    bool ok = _mm256_movemask_ps(bad) == 0;
    for (; i < n; ++i) {
        ok = ok && std::fabs(x[i]) <= 65504.0f;
        out[i] = fn::quant::f32_to_f16(x[i]);
    }
    return ok;
}

int64_t dn_act_bytes(fmt f) {
    return f == fmt::q2_0   ? int64_t(sizeof(act_q))
           : f == fmt::q5_1 ? int64_t(n_ff / 32) * int64_t(sizeof(block_q8_1))
                            : int64_t(n_ff / 32) * int64_t(sizeof(block_q8_0));
}
// ME6 rider: act_q's bit-plane image sizes the slot; its aligned loads take the slot's 64-byte alignment
static_assert(sizeof(act_q) <= dn_act_max && alignof(act_q) <= 64 && (n_ff / 32) * sizeof(block_q8_1) <= dn_act_max,
              "down activation");

void down_act(fmt f, const float * h, void * act) {
    if (f == fmt::q2_0) {
        quantize_act_q(h, n_ff, *static_cast<act_q *>(act));
    } else if (f == fmt::q5_1) {
        quantize_q8_1(h, act, n_ff);
    } else {
        quantize_q8_0(h, act, n_ff);
    }
}

void gate_up_tile(fmt f, const uint8_t * gate, const uint8_t * up, int64_t row_bytes, const block_q8_k * const * act,
                  int32_t nt, float * const * ff) {
    const size_t rb = size_t(row_bytes);
    const void * const * a = reinterpret_cast<const void * const *>(act);
    if (is_kq(f)) {
        kq_gu_rows(f, gate, up, rb, d_model, a, nt, ff, 0, 32);
    } else {
        iq_gu_rows(f, gate, up, rb, d_model, a, nt, ff, 0, 32);
    }
}

void down_rows(fmt f, const uint8_t * rows, int64_t row_bytes, int32_t n, const void * const * act, int32_t nt,
               float * const * out) {
    const size_t rb = size_t(row_bytes);
    if (f == fmt::q2_0) {
        q2_down_rows(rows, rb, n_ff / 64, reinterpret_cast<const act_q * const *>(act), nt, out, 0, n);
    } else if (f == fmt::iq4_nl) {
        iq4nl_down_rows(rows, rb, n_ff, act, nt, out, 0, n);
    } else {
        kq_down_rows(f, rows, rb, n_ff, act, nt, out, 0, n);
    }
}

int32_t window_meta(const int32_t * ids, int32_t rows, int32_t * e, int32_t * mask, int32_t * base, int32_t & n_slots) {
    int32_t n_u = 0;
    for (int32_t p = 0; p < rows * 10; ++p) {
        const int32_t id = ids[p];
        if (id < 0 || id >= 512) {
            continue;
        }
        int32_t u = 0;
        while (u < n_u && e[u] != id) {
            ++u;
        }
        if (u == n_u) {
            e[n_u] = id;
            mask[n_u] = 0;
            ++n_u;
        }
        mask[u] |= 1 << (p / 10);
    }
    n_slots = 0;
    for (int32_t u = 0; u < n_u; ++u) {
        base[u] = n_slots;
        n_slots += __builtin_popcount(uint32_t(mask[u]));
    }
    return n_u;
}

// ---- the pool (Strata pool.cpp) ----

namespace {

constexpr uint64_t pack_head(uint32_t epoch, uint32_t n, uint32_t i) {
    return (uint64_t(epoch) << 32) | (uint64_t(n) << 16) | uint64_t(i);
}

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

} // namespace

bool pool::start(const std::vector<int32_t> & cpus, std::string & err, int32_t park_ms) {
    stop();
    stop_.store(false);
    spin_before_sleep = std::chrono::milliseconds(std::max(0, park_ms));
    n_ = int32_t(cpus.size());
    parked_.store(0);
    pin_failed_.store(0);
    masks_ = std::vector<std::atomic<uint64_t>>(size_t(n_));
    ums_.assign(size_t(n_) + 1, unit_ms());
    host_tasks_ = 0;
    for (int32_t i = 0; i <= n_; ++i) { // the last the host's
        scratch_.push_back(static_cast<uint8_t *>(std::aligned_alloc(64, size_t(scratch_bytes))));
    }
    for (int32_t i = 0; i < n_; ++i) {
        threads_.emplace_back(&pool::worker, this, i, cpus[size_t(i)]);
    }
    // every worker at the park (Strata: parked_ == n_ from construction)
    const clk::time_point t0 = clk::now();
    while (parked_.load(std::memory_order_acquire) != uint32_t(n_)) {
        _mm_pause();
        if (clk::now() - t0 > stall) {
            err = "CPU pool: the workers did not start";
            return false;
        }
    }
    if (pin_failed_.load() != 0) {
        err = "CPU pool: a worker could not be pinned to its CPU";
        return false;
    }
    return true;
}

void pool::stop() {
    if (threads_.empty()) {
        return;
    }
    stop_.store(true, std::memory_order_release);
    publish(); // a parked worker notices the stop flag rather than sleeping through it
    for (auto & t : threads_) {
        t.join();
    }
    threads_.clear();
    for (uint8_t * p : scratch_) {
        std::free(p);
    }
    scratch_.clear();
    n_ = 0;
}

void pool::publish() {
    // Strata: both sides seq_cst (a worker going to sleep does sleepers_++ then reads epoch_; the host epoch_++ then
    // reads sleepers_): one of them sees the other's write, so no wakeup is lost
    epoch_.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) != 0) {
        std::lock_guard<std::mutex> lk(sleep_mu_);
        sleep_cv_.notify_all();
    }
}

void pool::worker(int32_t id, int32_t cpu) {
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
            pin_failed_.fetch_add(1);
        }
    }
    {
        cpu_set_t got;
        CPU_ZERO(&got);
        uint64_t mask = 0;
        if (pthread_getaffinity_np(pthread_self(), sizeof(got), &got) == 0) {
            for (int32_t c = 0; c < 64; ++c) {
                mask |= CPU_ISSET(c, &got) ? uint64_t(1) << c : 0;
            }
        }
        masks_[size_t(id)].store(mask);
    }
    uint32_t seen = 0;
    parked_.fetch_add(1, std::memory_order_acq_rel); // at the park before the first wait
    for (;;) {
        const clk::time_point parked_at = clk::now();
        uint32_t spins = 0;
        while (epoch_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_relaxed)) {
                return;
            }
            _mm_pause();
            if ((++spins & 1023u) != 0) {
                continue;
            }
            if (clk::now() - parked_at < spin_before_sleep) {
                continue;
            }
            std::unique_lock<std::mutex> lk(sleep_mu_);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            sleep_cv_.wait(lk, [&] {
                return epoch_.load(std::memory_order_seq_cst) != seen || stop_.load(std::memory_order_relaxed);
            });
            sleepers_.fetch_sub(1, std::memory_order_relaxed);
        }
        if (stop_.load(std::memory_order_acquire)) {
            return;
        }
        seen = epoch_.load(std::memory_order_acquire); // the batch this epoch published is visible from here
        parked_.fetch_sub(1, std::memory_order_acq_rel);
        drain(id, seen);
        parked_.fetch_add(1, std::memory_order_acq_rel);
    }
}

int32_t pool::claim(uint32_t epoch) {
    uint64_t h = head_.load(std::memory_order_acquire);
    for (;;) {
        if (uint32_t(h >> 32) != epoch) {
            return -1; // not the batch this thread woke for
        }
        const uint32_t n = uint32_t(h >> 16) & 0xffffu, i = uint32_t(h) & 0xffffu;
        if (i >= n) {
            return -1;
        }
        if (head_.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            return int32_t(i);
        }
    }
}

uint32_t pool::begin_batch(int32_t n) {
    // every job of the previous batch completed (wait_done) and none of it can be claimed now
    done_.store(0, std::memory_order_relaxed);
    const uint32_t e = epoch_.load(std::memory_order_relaxed) + 1; // only the host bumps the epoch
    head_.store(pack_head(e, uint32_t(n), 0), std::memory_order_release);
    publish();
    return e;
}

bool pool::wait_parked() {
    uint32_t spins = 0;
    clk::time_point t0{};
    while (parked_.load(std::memory_order_acquire) != uint32_t(n_)) {
        _mm_pause();
        if ((++spins & 1023u) != 0) {
            continue;
        }
        const clk::time_point now = clk::now();
        if (spins == 1024u) {
            t0 = now;
        } else if (now - t0 > stall) {
            return false;
        }
    }
    return true;
}

bool pool::wait_done(int32_t n) {
    uint32_t spins = 0, seen = 0;
    clk::time_point t0{};
    for (;;) {
        const uint32_t d = done_.load(std::memory_order_acquire);
        if (d >= uint32_t(n)) {
            return true;
        }
        _mm_pause();
        if ((++spins & 1023u) != 0) {
            continue;
        }
        const clk::time_point now = clk::now();
        if (spins == 1024u || d != seen) { // progress restarts the clock
            t0 = now;
            seen = d;
        } else if (now - t0 > stall) {
            return false;
        }
    }
}

int32_t pool::drain(int32_t id, uint32_t epoch) {
    uint8_t * const scratch = scratch_[size_t(id)];
    unit_ms & ms = ums_[size_t(id)];
    const bool timed = unit_timing_; // off in serving (ME2b amendment 1)
    int32_t claimed = 0;
    for (;;) {
        const int32_t i = claim(epoch);
        if (i < 0) {
            break;
        }
        ++claimed;
        // an equal range of the phase's units across all its experts (a range may span two)
        const int64_t g0 = int64_t(units_) * i / tasks_, g1 = int64_t(units_) * (i + 1) / tasks_;
        const int32_t per = mode_ == 1 ? n_tiles : n_pieces;
        for (int64_t u = g0; u < g1; ++u) {
            const int32_t e = int32_t(u / per), k = int32_t(u % per);
            const job & j = jobs_[e];
            float * o[job_max_rows];
            if (mode_ == 1) {
                const int32_t card = k / (n_ch / 32), g = k % (n_ch / 32);
                const block_q8_k * a[job_max_rows];
                for (int32_t t = 0; t < j.nt; ++t) {
                    a[t] = act_ + int64_t(j.row[t]) * n_q8k;
                    o[t] = h_.data() + (int64_t(e) * job_max_rows + t) * n_ff + card * n_ch + 32 * g;
                }
                if (j.ngate != nullptr) {
                    // ME4: the tile's native rows in place: gate and up rows 320 card + 32 g ..
                    const int64_t r0 = int64_t(card * n_ch + 32 * g) * j.ngu_row;
                    const clk::time_point t1 = timed ? clk::now() : clk::time_point();
                    gate_up_tile(gu_, j.ngate + r0, j.nup + r0, j.ngu_row, a, j.nt, o);
                    if (timed) {
                        ms.gu_compute += ms_since(t1);
                    }
                    continue;
                }
                const int64_t rb = gu_row_bytes(gu_);
                if (!timed) {
                    unpack_gate_up_tile(gu_, j.gu[card], g, scratch);
                    gate_up_tile(gu_, scratch, scratch + 32 * rb, rb, a, j.nt, o);
                    continue;
                }
                const clk::time_point t0 = clk::now();
                unpack_gate_up_tile(gu_, j.gu[card], g, scratch);
                const clk::time_point t1 = clk::now();
                gate_up_tile(gu_, scratch, scratch + 32 * rb, rb, a, j.nt, o);
                ms.gu_unpack += std::chrono::duration<double, std::milli>(t1 - t0).count();
                ms.gu_compute += ms_since(t1);
            } else {
                const int32_t n0 = k * down_piece;
                const void * a[job_max_rows];
                for (int32_t t = 0; t < j.nt; ++t) {
                    a[t] = a2_[size_t(e) * job_max_rows + t].b;
                    o[t] = j.out[t] + n0;
                }
                if (j.ndown != nullptr) {
                    // ME4: the piece's native rows in place (each row's 640 columns: both cards' halves)
                    const clk::time_point t1 = timed ? clk::now() : clk::time_point();
                    down_rows(dn_, j.ndown + int64_t(n0) * j.ndn_row, j.ndn_row, down_piece, a, j.nt, o);
                    if (timed) {
                        ms.dn_compute += ms_since(t1);
                    }
                    continue;
                }
                const int64_t rb = dn_row_bytes(dn_);
                if (!timed) {
                    unpack_down_rows(dn_, j.dn[0], j.dn[1], n0, down_piece, scratch);
                    down_rows(dn_, scratch, rb, down_piece, a, j.nt, o);
                    continue;
                }
                const clk::time_point t0 = clk::now();
                unpack_down_rows(dn_, j.dn[0], j.dn[1], n0, down_piece, scratch);
                const clk::time_point t1 = clk::now();
                down_rows(dn_, scratch, rb, down_piece, a, j.nt, o);
                ms.dn_unpack += std::chrono::duration<double, std::milli>(t1 - t0).count();
                ms.dn_compute += ms_since(t1);
            }
        }
        done_.fetch_add(1, std::memory_order_release);
    }
    return claimed;
}

// Strata's run_phase (pool.cpp, R2.2): publish, the host drains on its own scratch, then waits for the workers' tasks
// and for every worker back at the park (so the next batch's state can be written)
bool pool::run_phase(int32_t mode) {
    if (!wait_parked()) {
        return false;
    }
    mode_ = mode;
    const uint32_t e = begin_batch(tasks_);
    host_tasks_ += drain(n_, e);
    return wait_done(tasks_) && wait_parked();
}

bool pool::run(fmt gu, fmt dn, const block_q8_k * act, const job * jobs, int32_t n, run_stats & st) {
    if (n <= 0 || n_ == 0) {
        return n <= 0;
    }
    if (held_.exchange(true, std::memory_order_acquire)) {
        assert(!"experts cpu pool: a second caller inside run");
        std::fprintf(stderr, "experts cpu pool: a second caller inside run (refused)\n");
        return false;
    }
    struct hold {
        std::atomic<bool> & h;
        ~hold() { h.store(false, std::memory_order_release); }
    } const keep{held_};
    gu_ = gu;
    dn_ = dn;
    act_ = act;
    jobs_ = jobs;
    if (h_.size() < size_t(n) * job_max_rows * n_ff) {
        h_.resize(size_t(n) * job_max_rows * n_ff);
        a2_.resize(size_t(n) * job_max_rows);
    }
    for (unit_ms & m : ums_) {
        m = unit_ms();
    }
    const clk::time_point t0 = clk::now();
    units_ = n * n_tiles;
    tasks_ = std::min(3 * (n_ + 1), units_);
    if (!run_phase(1)) {
        return false;
    }
    const clk::time_point t1 = clk::now();
    for (int32_t e = 0; e < n; ++e) {
        for (int32_t t = 0; t < jobs[e].nt; ++t) {
            down_act(dn, h_.data() + (int64_t(e) * job_max_rows + t) * n_ff, a2_[size_t(e) * job_max_rows + t].b);
        }
    }
    const clk::time_point t2 = clk::now();
    units_ = n * n_pieces;
    tasks_ = std::min(3 * (n_ + 1), units_);
    if (!run_phase(2)) {
        return false;
    }
    // every worker's sums are visible: each wrote its own before its done_ release and park, both read after
    for (const unit_ms & m : ums_) {
        st.gu_unpack_ms += m.gu_unpack;
        st.gu_compute_ms += m.gu_compute;
        st.dn_unpack_ms += m.dn_unpack;
        st.dn_compute_ms += m.dn_compute;
    }
    st.gu_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    st.act_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    st.dn_ms += ms_since(t2);
    st.bytes += int64_t(n) * 2 * (gate_up_bytes(gu) + down_bytes(dn));
    return true;
}

} // namespace fn::experts::cpu
