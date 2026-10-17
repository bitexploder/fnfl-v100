#pragma once

// The prompt messages (PB1; FC-PF3's exchange rows, NC1 section 8): whole bodies copied card to card by the copy
// engines (PX1), not the decode rings' polled stores. A message slot is a 256-byte header, then the payload:
//   header: U32 gen_lo, gen_hi (the generation: chunk, layer, sublayer, half, kind; pgen), rows, kind
// The producer kernel writes the payload, then the header (pstamp, or the pack kernel's block 0 after its rows); the
// caller copies header and payload together (fn/model/prompt.cpp holds the events: producer done -> copy -> consumer,
// and the slot's reuse after its copy and its last reader); the consumer checks the header against the generation it
// was launched for and sets the error word's err_pmsg bit on a mismatch, writing nothing it would publish.
//
// Kernels:
//   ppack:    the sublayer's F32 partial [rows][2560] to the FP16 payload (round to nearest; |x| > 65,504 sets
//             err_prange), then the header.
//   pcombine: out[r][c] = f32(f16 card 0) + f32(f16 card 1) (both cards add in card order, so both hold the same bits;
//             the card's own partial is read from its own send slot, already rounded), then the streams
//             S[r][s][c] = fma(inject[r][s], out[r][c], S[r][s][c]) (FC-LAY1's combine, as decode's next HC forms it).
//   pseal:    (PB3) ppack's header alone, after a producer that wrote the FP16 payload itself with ppack's
//             arithmetic (pd_tc's and ep_sum's out16).
//   pstamp:   a header after the producer's payload (F32 messages: EMB, MOM, DMIX, PRED; SEL).
//   pcheck:   a header against its generation (consumers that read an F32 payload directly).
// Numerics: exact except the FP16 rounding (NC1 section 8; PX1's screen). Built without fast math.

#include <cstddef>
#include <cstdint>

namespace fn::xchg {

constexpr int64_t  pmsg_payload_at = 256;
constexpr uint32_t err_pmsg        = 1u << 25;  // a prompt message's header is not the generation its consumer expects
constexpr uint32_t err_prange = 1u << 26;  // a prompt FP16 operand out of range (an F32 value beyond 65,504 or NaN)

enum class pkind : int32_t { emb = 1, attn = 2, ffn = 3, sel = 4, mom = 5, dmix = 6, pred = 7 };

// the generation of a message: chunk (24 bits), layer (0..48: 48 the draft), sublayer (0, 1), half (0, 1), kind
uint64_t pgen(int64_t chunk, int32_t layer, int32_t sub, int32_t half, pkind kind);

inline int64_t pmsg_bytes(int64_t payload_bytes) {
    return pmsg_payload_at + payload_bytes;
}

bool ppack(int32_t rows, const float * partial, uint8_t * slot, uint64_t gen, uint32_t * err, void * stream);
bool pseal(int32_t rows, uint8_t * slot, uint64_t gen, void * stream);
bool pcombine(int32_t         rows,
              int32_t         card,
              const uint8_t * own_slot,
              const uint8_t * peer_slot,
              uint64_t        gen,
              const float *   inject,
              float *         streams,
              uint32_t *      err,
              void *          stream);
bool pstamp(uint8_t * slot, uint64_t gen, int32_t rows, pkind kind, void * stream);
bool pcheck(const uint8_t * slot, uint64_t gen, uint32_t * err, void * stream);

}  // namespace fn::xchg
