// The body of gate_up_kernel_u and (ME10) gate_up_kernel_u_pair (experts_gu.cuh), included inside each kernel so the
// body stays a kernel's own code (a __device__ body under two kernels changed the IQ2 instances' SASS, XB3 part 6).
// The including kernel defines X, FAM, B, DEPTH (its template), NB (its blocks a shard), blk (this block's index among
// them) and a (the block's shard arguments).
    if constexpr (X) { // ME1: a launch with nothing to run (gate_up_args::run_if)
        if (a.run_if != nullptr && *a.run_if == 0u) {
            return;
        }
    }
    constexpr int32_t C = cap_of(B); // SP2: the routing capacity (experts.h)
    constexpr bool U = true; // the generic sections' run-time form; F, FG and FU are unused placeholders
    constexpr fmt F = fmt::iq2_xxs;
    constexpr dfmt FG = dfmt::q4_k, FU = dfmt::q4_k;
    static_assert(!FAM::iq3 || B >= 4, "the IQ3 body: B = 4 and 5");
    // the routed format's sections: body(gu_fc<f>{}) for this launch's f
    const auto with_f = [&](auto && body) {
        if constexpr (!U) {
            body(gu_fc<F>{});
        } else if constexpr (FAM::iq3) {
            switch (static_cast<fmt>(a.u_f)) {
                case fmt::iq3_xxs:
                    body(gu_fc<fmt::iq3_xxs>{});
                    break;
                default:
                    body(gu_fc<fmt::iq3_s>{});
                    break;
            }
        } else {
            switch (static_cast<fmt>(a.u_f)) {
                case fmt::iq2_xxs:
                    body(gu_fc<fmt::iq2_xxs>{});
                    break;
                case fmt::iq2_xs:
                    body(gu_fc<fmt::iq2_xs>{});
                    break;
                default:
                    body(gu_fc<fmt::iq2_s>{});
                    break;
            }
        }
    };
    extern __shared__ uint4 smem_u4[];
    uint8_t * smem = reinterpret_cast<uint8_t *>(smem_u4);
    constexpr int32_t xs_off = U ? FAM::table + 2048 : gu_grid_len<F, FG, FU>();
    constexpr int32_t xm_off = xs_off + B * x_codes;
    constexpr int32_t red_off = xm_off + B * x_meta;
    constexpr int32_t meta_off = red_off + 2 * B * 64 * red_stride * 4;
    static_assert(xs_off % 16 == 0 && xm_off % 16 == 0 && red_off % 16 == 0 && meta_off % 16 == 0,
                  "16-byte aligned shared regions");
    const gu_smem_view<C> sv = {smem,
                             reinterpret_cast<uint32_t *>(smem + (U                         ? FAM::table
                                                                  : extra_iq3s<F, FG, FU>() ? gu_layout_of(F).grid_len
                                                                                            : 0)),
                             smem + xs_off,
                             reinterpret_cast<float4 *>(smem + xm_off),
                             reinterpret_cast<float *>(smem + red_off),
                             reinterpret_cast<meta_c<C> *>(smem + meta_off)};
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t lane = tid & 31;
    const int32_t w = tid >> 5;
    const int32_t j = tid >> 3;
    const int32_t rs = tid & 7;
    const uint32_t mz = gridDim.x == n_sm ? 0u : ~0u;

    // The grid tables and the rows' activations first, neither depends on the routing: coalesced 16-byte loads through
    // L1 (R3: written before this kernel and read by every SM; EX1's 4-byte L1-bypassing loads cost 7.7x the ideal L2
    // sectors). Thread i < 80 B takes run (b, j) = (i / 80, i % 80) of the rows. They land in shared memory after the
    // first weight chunks are issued.
    // the routed table (run-time data in the IQ2 body): a signed table is copied after the ring is primed
    const gu_layout gl = gu_layout_of(U ? static_cast<fmt>(a.u_f) : F);
    const bool ex_iq3s = U ? a.u_fg == int32_t(dfmt::iq3_s) || a.u_fu == int32_t(dfmt::iq3_s) : extra_iq3s<F, FG, FU>();
    const int32_t n_grid16 = gl.grid_len / 16;
    const int32_t n_sgrid16 = ex_iq3s ? 2048 / 16 : 0;
    static_assert(FAM::table / 16 + 2048 / 16 <= gu_threads, "one 16-byte load a thread");
    uint4 grid_v = {0, 0, 0, 0};
    if (tid < n_grid16) {
        grid_v = __ldg(reinterpret_cast<const uint4 *>(a.grids + gl.grid_off) + tid);
    } else if (tid < n_grid16 + n_sgrid16) {
        grid_v = __ldg(reinterpret_cast<const uint4 *>(a.grids + iq3s_off) + (tid - n_grid16));
    }
    const bool stager = tid < B * n_sub;
    // HC3: with X-HMIX, a stager whose run is in the peer's half polls it after the weight ring is primed (below)
    const bool peer_run = stager && a.hmix != nullptr && (tid % n_sub) / (n_sub / 2) == a.hmix_peer_half;
    uint4 xc0 = {0, 0, 0, 0}, xc1 = {0, 0, 0, 0};
    float2 xsc = {0.0f, 0.0f};
    if (stager && !peer_run) {
        const uint4 * src = reinterpret_cast<const uint4 *>(a.x.codes) + 2 * tid; // codes [B][2560]: run tid
        xc0 = __ldg(src);
        xc1 = __ldg(src + 1);
        xsc = __ldg(reinterpret_cast<const float2 *>(a.x.scales) + tid);
    }
    // XB3 part 1: warp 0 builds the routing meta (71b92a9's build_meta) and the block's units (block_units) while the
    // other warps' loads are in flight; block 0 writes the struct for down (the kernel boundary orders it before down's
    // read). The block's first unit comes from the order, so the ring's first loads follow the barrier.
    uint16_t * list = reinterpret_cast<uint16_t *>(smem + meta_off + gu_meta_bytes<C>);
    if (w == 0) {
        build_meta_of<B>([&](int32_t i) { return __ldg(a.ids + i); }, *sv.m);
        __syncwarp();
        if (sv.m->bad == 0) {
            block_units<X, C, NB>(*sv.m, a.cw, blk, list,
                        reinterpret_cast<uint16_t *>(a.meta + meta_order_of(C)), a.wt, a.skip);
        }
        if (blk == 0) {
            const uint4 * src = reinterpret_cast<const uint4 *>(sv.m);
#pragma unroll 1
            for (int32_t i = lane; i < meta_words_c<C>; i += 32) {
                reinterpret_cast<uint4 *>(a.meta)[i] = src[i];
            }
        }
    }
    __syncthreads();
    const meta_c<C> & m = *sv.m;
    if (m.bad != 0) {
        if (tid == 0 && blk == 0 && a.status != nullptr) {
            *a.status = 1u;
        }
        return;
    }
    if (list[gu_max_ranks<C, NB> + 1] == 0) { // no rank of the launch's 10 U + 10 is this block's
        return;
    }
    // the block's first routed unit (its tile primes the ring), or -1
    const int32_t t_first = list[gu_max_ranks<C, NB>] > 0 ? int32_t(list[0]) : -1;
    const int32_t sb = j >> 3;
    // byte offset of chunk q (0 = d, then aux when the format has one, then main) of this thread's record in format
    // C::f's tile; Q8_0's d chunk is the record's own (a d a sub-block), the others' their superblock's
    const auto chunk_off = [&](auto fc, int32_t q) -> int64_t {
        using C = decltype(fc);
        return q == 0 ? d_off(C::f) + (C::f == fmt::q8_0 ? tid : sb * 8 + rs) * 16
                      : (int64_t(q - 1 - C::NA) * gu_threads + tid) * 16;
    };
    // a chunk of format C::f's sequence: the aux chunk is one or two 8-byte loads, every other chunk one 16-byte load
    const auto load_chunk = [&](auto fc, const uint8_t * tl, int32_t q, uint32_t z) -> uint4 {
        using C = decltype(fc);
        if (C::NA == 1 && q == 1) {
            uint4 v = {0, 0, 0, 0};
            if (C::L.aux8) {
                const uint2 x = __ldcs(reinterpret_cast<const uint2 *>(tl + aux8_off(C::f) + tid * 8 + z));
                v.x = x.x;
                v.y = x.y;
            }
            if (C::L.sc8) {
                const uint2 x =
                    __ldcs(reinterpret_cast<const uint2 *>(tl + sc8_off(C::f) + ((j >> 1) * 8 + rs) * 8 + z));
                v.z = x.x;
                v.w = x.y;
            }
            return v;
        }
        return __ldcs(reinterpret_cast<const uint4 *>(tl + chunk_off(fc, q) + z));
    };
    const auto tile_of = [&](auto fc, int32_t t) -> const uint8_t * {
        const int32_t u = t / n_group;
        if constexpr (X) {
            return expert_base(a, m.e[u]) + (t - u * n_group) * decltype(fc)::T;
        }
        return a.w + (int64_t(m.e[u]) * n_group + (t - u * n_group)) * decltype(fc)::T;
    };

    // prime the ring: the first DEPTH chunks of the sequence (units are at least CC >= DEPTH chunks long)
    ring<DEPTH> rg;
    const uint8_t * tile = a.w;
    with_f([&](auto fc) {
        static_assert(decltype(fc)::CC >= DEPTH, "a unit has at least DEPTH chunks");
        if (t_first >= 0) {
            tile = tile_of(fc, t_first);
#pragma unroll
            for (int32_t q = 0; q < DEPTH; ++q) {
                rg.s[q] = load_chunk(fc, tile, q, 0u);
            }
        }
    });
    // X-HMIX's acquire (xchg.h): the peer half's runs, now that this block's first weight chunks are in flight
    if (peer_run) {
        fn::xchg::hmix_run(a.hmix, a.hmix_peer_half, tid / n_sub, tid % n_sub, xc0, xc1, xsc, a.err);
    }
    // the staged tables and activations into shared memory: the codes into DP2's swizzled chunks and each run's
    // {d, s} with the exact sums of its halves' codes (fn::dense::act, which the shared units read)
    if (tid < n_grid16) {
        reinterpret_cast<uint4 *>(sv.grid)[tid] = grid_v;
    } else if (tid < n_grid16 + n_sgrid16) {
        reinterpret_cast<uint4 *>(sv.sgrid)[tid - n_grid16] = grid_v;
    }
    if (stager) {
        const int32_t b = tid / n_sub, jj = tid - b * n_sub;
        int32_t slo = 0, shi = 0;
#pragma unroll
        for (int32_t t = 0; t < 4; ++t) {
            slo = fn::dense::dp4a(word_of(xc0, t), 0x01010101u, slo);
            shi = fn::dense::dp4a(word_of(xc1, t), 0x01010101u, shi);
        }
        uint4 * xs = reinterpret_cast<uint4 *>(sv.xs + b * x_codes);
        xs[fn::dense::swz(2 * jj)] = xc0;
        xs[fn::dense::swz(2 * jj + 1)] = xc1;
        sv.xm[tid] = make_float4(xsc.x, xsc.y, __int_as_float(slo), __int_as_float(shi)); // the sums' bits
    }
    __syncthreads();

    if (t_first >= 0) {
        // each thread's activations of sub-block j: the 8 lanes of a quarter-warp share j (a broadcast), and the four
        // quarter-warps' 16-byte reads fall in distinct bank groups (R6). SP4 H2: at B = 4 and 5 no row's fragment is
        // held a block (act's BA = 1, and neither array is read: row_out_c reads each row's at its turn; da keeps B for
        // the Q8_0 section's row_out_q8 signature, which this body never runs)
        constexpr int32_t BA = B <= 3 ? B : 1;
        [[maybe_unused]] uint32_t act[BA][8];
        [[maybe_unused]] float da[B];
        if constexpr (B <= 3) {
#pragma unroll
            for (int32_t b = 0; b < B; ++b) {
                const uint4 * xb = reinterpret_cast<const uint4 *>(sv.xs + b * x_codes);
                const uint4 c0 = xb[fn::dense::swz(2 * j)], c1 = xb[fn::dense::swz(2 * j + 1)];
                act[b][0] = c0.x;
                act[b][1] = c0.y;
                act[b][2] = c0.z;
                act[b][3] = c0.w;
                act[b][4] = c1.x;
                act[b][5] = c1.y;
                act[b][6] = c1.z;
                act[b][7] = c1.w;
                da[b] = sv.xm[b * n_sub + j].x;
            }
        }

        // the block's routed unit i (10 u + g), -1 past the last: a volatile load at each use, so no entry is held in
        // a register across a unit (the loops hold the index alone, as they held the unit before XB2; IQ2_S at B = 4
        // spilled at the 96-register cap otherwise)
        const volatile uint16_t * lv = list;
        const int32_t n_mine = list[gu_max_ranks<C, NB>];
        const auto unit_at = [&](int32_t i) -> int32_t { return i < n_mine ? int32_t(lv[i]) : -1; };

        with_f([&](auto fc) {
            using C = decltype(fc);
            // The block's routed unit i, then its next one (none: -1). The ring slot of chunk q is (P + q) % DEPTH, a
            // constant: data in flight is never moved between registers (a move waits for its load; EX1 v1 lost the
            // stream depth that way). The refill of the last chunks targets the next routed unit (phase (P + C::CC) %
            // DEPTH), or the current tile again after the block's last one, so no refill is predicated.
            const auto run_unit = [&](auto phase, int32_t i, int32_t buf) {
                constexpr int32_t P = decltype(phase)::value;
                const uint32_t mask = m.mask[unit_at(i) / n_group];
                const int32_t tn = unit_at(i + 1);
                const uint8_t * tile_next = tn >= 0 ? tile_of(fc, tn) : tile;
                float * red_buf = sv.red + buf * B * 64 * red_stride;

                // the d chunk: rows 0..7's d, converted where the chunk is taken (XB2 (c): held as words, it compiled
                // to a copy right after the next unit's load, a wait long before the data was used)
                float dh8[8];
                // the aux chunk's aux8 bytes (x, y) and sc8 bytes (z, w) of rows 0..7; IQ2_S: its qh (x, y) and scales
                // (z, w) chunk, main chunk 0
                [[maybe_unused]] bytes8 ax8 = {0, 0}, sc8 = {0, 0};
                [[maybe_unused]] uint4 held = {0, 0, 0, 0}; // IQ3: the sign words waiting
                [[maybe_unused]] int32_t is8[B];            // Q8_0: a row's integer sums, over its two chunks
                // the unit's chunks in order; out(rd, r) finishes row r of a record (Q8_0 finishes its rows itself)
                const auto chunks = [&](auto out) {
#pragma unroll
                    for (int32_t q = 0; q < C::CC; ++q) {
                        const int32_t sl = (P + q) % DEPTH;
                        const uint4 v = rg.s[sl];
                        // refill the slot just taken: chunk q + DEPTH of the sequence, in this unit or the next
                        const int32_t qn = q + DEPTH;
                        const uint32_t z = order_after(v.x, mz);
                        rg.s[sl] = qn < C::CC ? load_chunk(fc, tile, qn, z) : load_chunk(fc, tile_next, qn - C::CC, z);
                        if (q == 0) {
#pragma unroll
                            for (int32_t mr = 0; mr < 8; ++mr) {
                                dh8[mr] = (mr & 1) ? half_hi(word_of(v, mr >> 1)) : half_lo(word_of(v, mr >> 1));
                            }
                            continue;
                        }
                        if (C::NA == 1 && q == 1) {
                            ax8 = take8(v.x, v.y);
                            sc8 = take8(v.z, v.w);
                            continue;
                        }
                        const int32_t c = q - 1 - C::NA;
                        if constexpr (C::f == fmt::q8_0) {
                            // chunk c: half c & 1 of row c / 2's sub-block, against codes 16 (c & 1) .. of each routed
                            // row
                            const int32_t mrow = c >> 1, hh = c & 1;
#pragma unroll
                            for (int32_t b = 0; b < B; ++b) {
                                if (((mask >> b) & 1u) == 0) {
                                    continue;
                                }
                                int32_t acc = hh == 0 ? 0 : is8[b];
#pragma unroll
                                for (int32_t i = 0; i < 4; ++i) {
                                    acc = __dp4a(static_cast<int32_t>(word_of(v, i)),
                                                 static_cast<int32_t>(act[b][4 * hh + i]), acc);
                                }
                                is8[b] = acc;
                            }
                            if (hh == 1) {
                                row_out_q8<B>(is8, dh8[mrow], da, mask, red_buf, rs + 8 * mrow, w, lane);
                            }
                            continue;
                        }
                        row_dec rd;
                        if constexpr (C::f == fmt::iq2_xxs || C::f == fmt::iq2_xs) {
                            [[maybe_unused]] const uint2 * grid = reinterpret_cast<const uint2 *>(sv.grid); // IQ2_XS
#pragma unroll
                            for (int32_t h = 0; h < 2; ++h) {
                                const int32_t mrow = 2 * c + h;
                                const uint2 qq = h == 0 ? make_uint2(v.x, v.y) : make_uint2(v.z, v.w);
                                const float dh = dh8[mrow];
                                if constexpr (C::f == fmt::iq2_xxs) {
                                    dec_iq2_xxs(rd, qq, dh, grid);
                                    out(rd, rs + 8 * mrow);
                                } else {
                                    const uint32_t sc = byte8(ax8, mrow);
                                    dec_iq2_xs(rd, qq, sc, dh, grid);
                                    out(rd, rs + 8 * mrow);
                                }
                            }
                        } else if constexpr (C::f == fmt::iq2_s) {
                            if (c == 0) {
                                ax8 = take8(v.x, v.y); // qh of rows 0..7
                                sc8 = take8(v.z, v.w); // scales of rows 0..7
                                continue;
                            }
                            const uint2 * grid = reinterpret_cast<const uint2 *>(sv.grid);
#pragma unroll
                            for (int32_t h = 0; h < 2; ++h) {
                                const int32_t mrow = 2 * (c - 1) + h;
                                const float dh = dh8[mrow];
                                const uint32_t qh = byte8(ax8, mrow);
                                const uint32_t sc = byte8(sc8, mrow);
                                dec_iq2_s(rd, h == 0 ? v.x : v.z, h == 0 ? v.y : v.w, qh, sc, dh, grid);
                                out(rd, rs + 8 * mrow);
                            }
                        } else {
                            // IQ3: chunk k of row quad hq: k 0 [q r0][sg r0][sg r1], k 1 [q r1][sg r2][sg r3],
                            // k 2 [q r2][q r3]
                            const uint32_t * grid = reinterpret_cast<const uint32_t *>(sv.grid);
                            const int32_t hq = c / 3;
                            const int32_t k = c - 3 * hq;
                            const auto row = [&](int32_t mrow, uint2 qq, uint32_t sg) {
                                const float dh = dh8[mrow];
                                if constexpr (C::f == fmt::iq3_xxs) {
                                    dec_iq3_xxs(rd, qq, sg, dh, grid);
                                } else {
                                    const uint32_t qh = byte8(ax8, mrow);
                                    const uint32_t scb = byte8(sc8, mrow);
                                    dec_iq3_s(rd, qq, sg, qh, (j & 1) ? scb >> 4 : scb & 0xfu, dh, grid);
                                }
                                out(rd, rs + 8 * mrow);
                            };
                            if (k == 0) {
                                held.x = v.w;
                                row(4 * hq + 0, make_uint2(v.x, v.y), v.z);
                            } else if (k == 1) {
                                held.y = v.z;
                                held.z = v.w;
                                row(4 * hq + 1, make_uint2(v.x, v.y), held.x);
                            } else {
                                row(4 * hq + 2, make_uint2(v.x, v.y), held.y);
                                row(4 * hq + 3, make_uint2(v.z, v.w), held.z);
                            }
                        }
                    }
                };
                // XB3 K1b: one body. Every record's rows run the unit's first routed row b0 with no row test (row_out1
                // on b0's codes and d, read once a unit from shared memory by b0's address, never act[b0]), then the
                // other routed rows under one unit-uniform test, inside which today's per-row guard runs over the rest.
                // Each row's partial is today's, in today's row order. Q8_0 (the draft's) keeps its own loop. At B = 4
                // the rows after b0 are read a unit too, into slots k = row - b0 - 1 (row_out over B - 1 slots, its
                // partials from row b0 + 1 on), instead of today's act held a block: act1 beside all four rows spilled
                // at 96 registers (IQ2_S, IQ3_S).
                if constexpr (C::f == fmt::q8_0) {
                    chunks([](const row_dec &, int32_t) {});
                } else {
                    constexpr bool SPLIT = C::f == fmt::iq2_xs || C::f == fmt::iq2_s;
                    const auto row_act = [&](int32_t b, uint32_t (&o)[8], float & d) {
                        const uint4 * xb = reinterpret_cast<const uint4 *>(sv.xs + b * x_codes);
                        const uint4 c0 = xb[fn::dense::swz(2 * j)], c1 = xb[fn::dense::swz(2 * j + 1)];
                        o[0] = c0.x;
                        o[1] = c0.y;
                        o[2] = c0.z;
                        o[3] = c0.w;
                        o[4] = c1.x;
                        o[5] = c1.y;
                        o[6] = c1.z;
                        o[7] = c1.w;
                        d = sv.xm[b * n_sub + j].x;
                    };
                    const int32_t b0 = __ffs(static_cast<int32_t>(mask)) - 1;
                    uint32_t act1[8];
                    float da1;
                    row_act(b0, act1, da1);
                    if constexpr (B >= 4) {
                        // SP4 H2: b0 by row_out1, then the unit's other routed rows (the mask without b0, bit b: row
                        // b) through row_out_c's one compact loop, at every decoded row
                        float * red1 = red_buf + b0 * 64 * red_stride;
                        const uint32_t rest = mask & (mask - 1u);
                        chunks([&](const row_dec & rd, int32_t r) {
                            row_out1<SPLIT>(rd, act1, da1, red1, r, w, lane);
                            row_out_c<SPLIT>(rd, sv.xs, sv.xm, j, rest, red_buf, r, w, lane);
                        });
                    } else {
                        float * red1 = red_buf + b0 * 64 * red_stride;
                        const uint32_t rest = mask & (mask - 1u);
                        chunks([&](const row_dec & rd, int32_t r) {
                            row_out1<SPLIT>(rd, act1, da1, red1, r, w, lane);
                            if (rest != 0u) {
                                row_out<SPLIT, B>(rd, act, da, rest, red_buf, r, w, lane);
                            }
                        });
                    }
                }
                tile = tile_next;
                __syncthreads();
                // epilogue of the unit: warp b (a routed row) sums its 20 warp partials per channel, GLU, FNFL's format
                if (w < B && ((mask >> w) & 1u) != 0) {
                    const int32_t t = unit_at(i);
                    const int32_t u = t / n_group;
                    glu_out<gu_warps, red_stride>(a, red_buf + w * 64 * red_stride, slot_of(m, u, w), t - u * n_group,
                                                  lane);
                }
            };
            // the ring's phase is a compile-time constant (R12): when a unit's chunks are a multiple of DEPTH every
            // unit starts at phase 0, so one copy of the unit body serves them all (EX2 B3: the executed code a launch
            // fetches cold, M9); an odd C::CC takes two units a trip, one copy a phase. IQ2_S at B = 4 keeps two
            // copies: its single copy's runtime buffer index spilled 8 bytes at the 96-register cap (the per-triple
            // kernel's B = 4; this body ran B = 1-3 only before SP4 H2, whose B = 4 and 5 keep one copy).
            static_assert((2 * C::CC) % DEPTH == 0, "the phase repeats every two units");
            if constexpr (C::CC % DEPTH == 0) {
#pragma unroll 1
                for (int32_t i = 0; i < n_mine; ++i) {
                    run_unit(std::integral_constant<int32_t, 0>{}, i, i & 1);
                }
            } else {
                for (int32_t i = 0; i < n_mine; i += 2) {
                    run_unit(std::integral_constant<int32_t, 0>{}, i, 0);
                    if (i + 1 < n_mine) {
                        run_unit(std::integral_constant<int32_t, C::CC % DEPTH>{}, i + 1, 1);
                    }
                }
            }
        });
    }
    // the shared expert's pair: body(gate, up) as integral constants, this launch's (the IQ2 body: each pair's code
    // behind one block-uniform switch, the ten pairs the IQ2 layers meet; the launcher admits only deployed triples)
    const auto with_pair = [&](auto && body) {
        using std::integral_constant;
        if constexpr (!U) {
            body(integral_constant<dfmt, FG>{}, integral_constant<dfmt, FU>{});
        } else {
#define FN_GU_PAIR(g, u)                                                                                               \
    case int32_t(dfmt::g) * 16 + int32_t(dfmt::u):                                                                     \
        body(integral_constant<dfmt, dfmt::g>{}, integral_constant<dfmt, dfmt::u>{});                                  \
        break;
            switch (a.u_fg * 16 + a.u_fu) {
                FN_GU_PAIR(iq4_xs, iq3_s)
                FN_GU_PAIR(q4_k, q4_k)
                FN_GU_PAIR(iq4_xs, q6_k)
                FN_GU_PAIR(iq3_s, iq4_xs)
                FN_GU_PAIR(iq3_s, iq3_s)
                FN_GU_PAIR(q4_k, iq3_s)
                FN_GU_PAIR(q4_k, q5_k)
                FN_GU_PAIR(q5_k, q4_k)
                FN_GU_PAIR(iq4_xs, q4_k)
                FN_GU_PAIR(iq4_xs, iq4_xs)
                default:
                    break;
            }
#undef FN_GU_PAIR
        }
    };
    // the block's shared ranks after its routed units, in rank order: two when the shared class straddles a snake turn
    // (XB2 Amendment 1). Each takes the whole partials region, so the unit before it must have finished its epilogue.
#pragma unroll 1
    for (int32_t i = list[gu_max_ranks<C, NB>]; i < list[gu_max_ranks<C, NB> + 1]; ++i) {
        const int32_t sh_g = list[i] - order_shared_of(C);
        if (i > 0) {
            __syncthreads();
        }
        const fn::dense::act x = {sv.xs, sv.xm, D};
        // the thread's warp and lane read again (opaque), not held across the routed units (K1b at B = 4: IQ3_XXS
        // spilled one register here)
        const int32_t tx = opaque(static_cast<int32_t>(threadIdx.x));
        const int32_t sw = tx >> 5, sl = tx & 31;
        if constexpr (B <= 3) {
            with_pair([&](auto gc, auto uc) {
                shared_unit<decltype(gc)::value, decltype(uc)::value, B>(a, sh_g, sw, sl, x, sv.sgrid, sv.red);
            });
        } else {
            // SP4 H2: a direct switch (through with_pair ptxas outlined the pair at B = 4 and 5 in SP4's kernel)
#define FN_GU_PAIR(g, u)                                                                                               \
    case int32_t(dfmt::g) * 16 + int32_t(dfmt::u):                                                                     \
        shared_unit<dfmt::g, dfmt::u, B>(a, sh_g, sw, sl, x, sv.sgrid, sv.red);                                        \
        break;
            if constexpr (FAM::iq3) {
                // XB3 C1: the eight pairs the IQ3 layers meet (experts_inst.h)
                switch (a.u_fg * 16 + a.u_fu) {
                    FN_GU_PAIR(iq4_xs, q6_k)
                    FN_GU_PAIR(iq3_s, iq3_s)
                    FN_GU_PAIR(q4_k, iq3_s)
                    FN_GU_PAIR(iq3_s, q4_k)
                    FN_GU_PAIR(q6_k, iq3_s)
                    FN_GU_PAIR(iq4_xs, iq4_xs)
                    FN_GU_PAIR(iq4_xs, q4_k)
                    FN_GU_PAIR(q4_k, iq4_xs)
                    default:
                        break;
                }
            } else {
                switch (a.u_fg * 16 + a.u_fu) {
                    FN_GU_PAIR(iq4_xs, iq3_s)
                    FN_GU_PAIR(q4_k, q4_k)
                    FN_GU_PAIR(iq4_xs, q6_k)
                    FN_GU_PAIR(iq3_s, iq4_xs)
                    FN_GU_PAIR(iq3_s, iq3_s)
                    FN_GU_PAIR(q4_k, iq3_s)
                    FN_GU_PAIR(q4_k, q5_k)
                    FN_GU_PAIR(q5_k, q4_k)
                    FN_GU_PAIR(iq4_xs, q4_k)
                    FN_GU_PAIR(iq4_xs, iq4_xs)
                    default:
                        break;
                }
            }
#undef FN_GU_PAIR
        }
        __syncthreads();
        if (sw < B) {
            glu_out<sh_parts, sh_stride>(a, sv.red + sw * 64 * sh_stride, m.n_slots + sw, sh_g, sl);
        }
    }
