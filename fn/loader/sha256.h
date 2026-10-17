#pragma once

// SHA-256 (FIPS 180-4) for the loader's placement check (R1): the x86 SHA extensions when the CPU has them, otherwise
// a portable path. No third-party code.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace fn::loader {

using digest = std::array<uint8_t, 32>;

class sha256 {
  public:
    explicit sha256(bool portable = false); // portable: never the SHA extensions (the self-test)
    void update(const void * data, size_t n);
    digest finish(); // the object must not be updated afterwards

  private:
    bool hw_ = false;
    uint32_t state_[8] = {};
    uint8_t buf_[64] = {};
    size_t n_buf_ = 0;
    uint64_t n_total_ = 0;
};

std::string hex(const digest & d);

// the FIPS test vectors on both paths; false (and a message on stderr) on any miss
bool sha256_self_test();

// true when the SHA-extension path is in use
bool sha256_hardware();

} // namespace fn::loader
