#pragma once

// SM1: the sampler's random numbers (SM0 section 6; FC-DEC2): Philox 4x32-10 (Salmon et al., "Parallel random numbers:
// as easy as 1, 2, 3", SC 2011; the constants are the paper's), keyed by (seed, request, position, purpose), so both
// cards and the CPU reference draw the same numbers, and a rolled-back position redraws the same ones. Host and device:
// the CUDA compiler sees __host__ __device__; a host compiler sees plain inline functions.
//
// key   = (seed low 32 bits, seed high 32 bits)
// count = (position, purpose, request low 32 bits, request high 32 bits)
// The four output words give two 64-bit uniforms: words 0 and 1 (u64 = w0 << 32 | w1), then words 2 and 3.

#include <cstdint>

#if defined(__CUDACC__)
#define FN_SAMP_HD __host__ __device__
#else
#define FN_SAMP_HD
#endif

namespace fn::samp {

// the purposes of a position's uniforms: every use draws from its own counter, so no two uses share a number
enum purpose : uint32_t {
    p_shard = 1,       // the shard a draw takes (full path), or the draw over the kept list (top-k path)
    p_local = 2,       // a card's local candidate inside its kept part (full path)
    p_accept = 3,      // the accept test of a verify row (SM0 section 5)
    p_resid_shard = 4, // the residual's shard (full path), or the residual draw over the kept list (top-k path)
    p_resid_local = 5, // a card's local residual candidate (full path)
    p_draft = 6,       // the draft's pick (SM0 section 5, last item)
};

struct philox_out {
    uint32_t w[4];
};

FN_SAMP_HD inline void philox_mul(uint32_t a, uint32_t b, uint32_t & hi, uint32_t & lo) {
    const uint64_t p = static_cast<uint64_t>(a) * b;
    hi = static_cast<uint32_t>(p >> 32);
    lo = static_cast<uint32_t>(p);
}

FN_SAMP_HD inline philox_out philox4x32_10(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0,
                                           uint32_t k1) {
    constexpr uint32_t m0 = 0xD2511F53u, m1 = 0xCD9E8D57u; // multipliers
    constexpr uint32_t w0 = 0x9E3779B9u, w1 = 0xBB67AE85u; // Weyl key increments
    for (int32_t r = 0; r < 10; ++r) {
        uint32_t hi0 = 0, lo0 = 0, hi1 = 0, lo1 = 0;
        philox_mul(m0, c0, hi0, lo0);
        philox_mul(m1, c2, hi1, lo1);
        const uint32_t n0 = hi1 ^ c1 ^ k0;
        const uint32_t n2 = hi0 ^ c3 ^ k1;
        c0 = n0;
        c1 = lo1;
        c2 = n2;
        c3 = lo0;
        k0 += w0;
        k1 += w1;
    }
    philox_out o = {{c0, c1, c2, c3}};
    return o;
}

// the two 64-bit uniforms of (seed, request, position, purpose)
FN_SAMP_HD inline void uniforms(uint64_t seed, uint64_t request, uint32_t position, uint32_t purpose, uint64_t & u0,
                                uint64_t & u1) {
    const philox_out o =
        philox4x32_10(position, purpose, static_cast<uint32_t>(request), static_cast<uint32_t>(request >> 32),
                      static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
    u0 = (static_cast<uint64_t>(o.w[0]) << 32) | o.w[1];
    u1 = (static_cast<uint64_t>(o.w[2]) << 32) | o.w[3];
}

// u in [0, 1) as a double: the top 53 bits of a 64-bit uniform (exact)
FN_SAMP_HD inline double unit_double(uint64_t u) {
    return static_cast<double>(u >> 11) * (1.0 / 9007199254740992.0);
}

// floor(u * total / 2^64): the integer target of a draw over integer masses summing to total (exact; below total)
FN_SAMP_HD inline uint64_t scaled_target(uint64_t u, uint64_t total) {
#if defined(__CUDA_ARCH__)
    // u * total = (uh * 2^32 + ul) * total; the high 64 bits of the 128-bit product
    return __umul64hi(u, total);
#else
    return static_cast<uint64_t>((static_cast<unsigned __int128>(u) * total) >> 64);
#endif
}

} // namespace fn::samp

#undef FN_SAMP_HD
