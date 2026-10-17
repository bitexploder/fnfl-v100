#include "fn/loader/sha256.h"

#include <cstdio>
#include <cstring>
#include <immintrin.h>

namespace fn::loader {

namespace {

alignas(16) constexpr uint32_t k_round[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
};

constexpr uint32_t k_init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

inline uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

void compress_portable(uint32_t s[8], const uint8_t * p, size_t blocks) {
    for (size_t b = 0; b < blocks; ++b, p += 64) {
        uint32_t w[64];
        for (int t = 0; t < 16; ++t) {
            w[t] = (uint32_t(p[4 * t]) << 24) | (uint32_t(p[4 * t + 1]) << 16) | (uint32_t(p[4 * t + 2]) << 8) |
                   uint32_t(p[4 * t + 3]);
        }
        for (int t = 16; t < 64; ++t) {
            const uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
            const uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16] + s0 + w[t - 7] + s1;
        }
        uint32_t a = s[0], b2 = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
        for (int t = 0; t < 64; ++t) {
            const uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k_round[t] + w[t];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b2) ^ (a & c) ^ (b2 & c));
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b2;
            b2 = a;
            a = t1 + t2;
        }
        s[0] += a;
        s[1] += b2;
        s[2] += c;
        s[3] += d;
        s[4] += e;
        s[5] += f;
        s[6] += g;
        s[7] += h;
    }
}

// The SHA extensions: the message schedule of a block first (msg1/msg2), then 16 groups of four rounds (rnds2 twice).
// The state is kept as ABEF and CDGH, as the instructions take it.
__attribute__((target("sha,sse4.1,ssse3"))) void compress_shani(uint32_t s[8], const uint8_t * p, size_t blocks) {
    const __m128i bswap = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&s[0])); // reason: SIMD load of the state words
    __m128i st1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&s[4]));
    tmp = _mm_shuffle_epi32(tmp, 0xB1);          // CDAB
    st1 = _mm_shuffle_epi32(st1, 0x1B);          // EFGH
    __m128i st0 = _mm_alignr_epi8(tmp, st1, 8);  // ABEF
    st1 = _mm_blend_epi16(st1, tmp, 0xF0);       // CDGH
    for (size_t b = 0; b < blocks; ++b, p += 64) {
        const __m128i abef = st0, cdgh = st1;
        __m128i w[16];
        for (int g = 0; g < 4; ++g) {
            w[g] = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 16 * g)), bswap);
        }
        for (int g = 4; g < 16; ++g) {
            w[g] = _mm_sha256msg2_epu32(
                _mm_add_epi32(_mm_sha256msg1_epu32(w[g - 4], w[g - 3]), _mm_alignr_epi8(w[g - 1], w[g - 2], 4)),
                w[g - 1]);
        }
        for (int g = 0; g < 16; ++g) {
            __m128i m = _mm_add_epi32(w[g], _mm_load_si128(reinterpret_cast<const __m128i *>(k_round + 4 * g)));
            st1 = _mm_sha256rnds2_epu32(st1, st0, m);
            m = _mm_shuffle_epi32(m, 0x0E);
            st0 = _mm_sha256rnds2_epu32(st0, st1, m);
        }
        st0 = _mm_add_epi32(st0, abef);
        st1 = _mm_add_epi32(st1, cdgh);
    }
    tmp = _mm_shuffle_epi32(st0, 0x1B);          // FEBA
    st1 = _mm_shuffle_epi32(st1, 0xB1);          // DCHG
    st0 = _mm_blend_epi16(tmp, st1, 0xF0);       // DCBA
    st1 = _mm_alignr_epi8(st1, tmp, 8);          // ABEF -> HGFE
    _mm_storeu_si128(reinterpret_cast<__m128i *>(&s[0]), st0);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(&s[4]), st1);
}

bool has_shani() {
    static const bool yes = __builtin_cpu_supports("sha") && __builtin_cpu_supports("sse4.1");
    return yes;
}

void compress(bool hw, uint32_t s[8], const uint8_t * p, size_t blocks) {
    if (hw) {
        compress_shani(s, p, blocks);
    } else {
        compress_portable(s, p, blocks);
    }
}

} // namespace

sha256::sha256(bool portable) : hw_(!portable && has_shani()) {
    std::memcpy(state_, k_init, sizeof(state_));
}

void sha256::update(const void * data, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    n_total_ += n;
    if (n_buf_ > 0) {
        const size_t take = n < 64 - n_buf_ ? n : 64 - n_buf_;
        std::memcpy(buf_ + n_buf_, p, take);
        n_buf_ += take;
        p += take;
        n -= take;
        if (n_buf_ == 64) {
            compress(hw_, state_, buf_, 1);
            n_buf_ = 0;
        }
    }
    if (n >= 64) {
        compress(hw_, state_, p, n / 64);
        p += n / 64 * 64;
        n %= 64;
    }
    if (n > 0) {
        std::memcpy(buf_, p, n);
        n_buf_ = n;
    }
}

digest sha256::finish() {
    const uint64_t bits = n_total_ * 8;
    uint8_t pad[72] = {0x80};
    const size_t n_pad = (n_buf_ < 56 ? 56 : 120) - n_buf_;
    for (int i = 0; i < 8; ++i) {
        pad[n_pad + i] = uint8_t(bits >> (56 - 8 * i));
    }
    update(pad, n_pad + 8);
    digest d;
    for (int i = 0; i < 8; ++i) {
        d[4 * i] = uint8_t(state_[i] >> 24);
        d[4 * i + 1] = uint8_t(state_[i] >> 16);
        d[4 * i + 2] = uint8_t(state_[i] >> 8);
        d[4 * i + 3] = uint8_t(state_[i]);
    }
    return d;
}

std::string hex(const digest & d) {
    static const char * digits = "0123456789abcdef";
    std::string s(64, '0');
    for (size_t i = 0; i < 32; ++i) {
        s[2 * i] = digits[d[i] >> 4];
        s[2 * i + 1] = digits[d[i] & 15];
    }
    return s;
}

bool sha256_hardware() {
    return has_shani();
}

bool sha256_self_test() {
    struct vec {
        const char * msg;
        size_t repeat;
        const char * want;
    };
    static const vec vecs[] = {
        {"", 1, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", 1, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 1,
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {"a", 1000000, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"},
    };
    bool ok = true;
    for (int pass = 0; pass < 2; ++pass) {
        for (const vec & v : vecs) {
            sha256 h(pass == 1);
            const size_t len = std::strlen(v.msg);
            for (size_t i = 0; i < v.repeat; ++i) {
                h.update(v.msg, len);
            }
            const std::string got = hex(h.finish());
            if (got != v.want) {
                fprintf(stderr, "sha256 self-test (%s path): '%.16s' x %zu gives %s, want %s\n",
                        pass == 0 ? "default" : "portable", v.msg, v.repeat, got.c_str(), v.want);
                ok = false;
            }
        }
    }
    return ok;
}

} // namespace fn::loader
