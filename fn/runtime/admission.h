#pragma once

// ME9 (retain useful expert copies): the admission records, CPU only, no device. Up to per_layer reserved slots a
// target layer; a lane miss the GPU already copies lands in one of them instead of the scratch lane, and becomes
// resident at the next publication boundary. This unit holds the records, their states and generations, and the two
// choices (which residents to retire, which misses to admit); the expert cache owns one table, the residency owns the
// published slot/key maps, and the service copies. The states (tickets/ME9-retain-lane-experts.md):
//   none     -> empty:    reserve (a resident retired at a boundary, or an orphaned slot adopted); new generation
//   empty    -> copying:  start (the service, after the slot's copy and its completion event were both enqueued)
//   copying  -> landed:   land (the boundary saw the record's event complete, the generation current)
//   landed   -> none:     publish (the boundary installed the key in the slot; the slot leaves the table)
//   copying | landed -> empty: invalidate (a failure, an abandoned pass, released waits; new generation, no key)
//   any      -> none:     discard (reset to the boot placement: the slot handed back to the caller)
// A reserved slot survives a prompt loan: the loan may borrow its bytes, which are garbage when it returns them
// (return_from_loan: a new generation); nothing is copied into a reserved slot while a loan is taken or held, a
// refill is active, or an adapt pass is pending (the round's immutable eligibility, decided at round_start).

#include <cstdint>
#include <vector>

namespace fn::runtime::adm {

constexpr int32_t per_layer = 3;
constexpr int32_t n_layers = 48;
constexpr int32_t max_recs = n_layers * per_layer; // 144

enum class state : uint8_t { none = 0, empty, copying, landed };

struct rec {
    int32_t slot = -1;  // the reserved slot (none: -1)
    int32_t layer = -1; // its layer (the slot's class holds this layer's entries)
    int32_t key = -1;   // copying, landed: the admitted key (layer x 512 + expert)
    uint32_t gen = 0;   // the generation the copy started in
    state st = state::none;
};

// a resident the boundary may retire: its slot, key and decayed usage
struct cand {
    int32_t slot = -1, key = -1;
    float usage = 0.0f;
};

// a GPU-served miss of a window: its index u in the window, its row count and route order (first appearance)
struct miss {
    int32_t u = -1, rows = 0, order = 0;
};

// why a round admits nothing (the counters' split)
enum class off : uint8_t { none = 0, mode, devices, loan, refill, pass, failed };

class table {
  public:
    table() = default;
    // the records of layer l: indices l x per_layer .. + per_layer - 1
    rec & at(int32_t i) { return r_[size_t(i)]; }
    const rec & at(int32_t i) const { return r_[size_t(i)]; }
    int32_t held(int32_t layer) const; // records of the layer not none
    int32_t held_all() const;
    int32_t in_state(state s) const;
    // the record reserving slot s, -1
    int32_t of_slot(int32_t s) const;
    // a record copying or landed with key (a key admitted once a pass: a repeat miss does not take a second slot)
    bool in_flight(int32_t key) const;

    // none -> empty: slot s of layer l reserved (false: the layer's records are full, or s is already reserved)
    bool reserve(int32_t layer, int32_t s);
    // empty -> copying (the service); false otherwise
    bool start(int32_t i, int32_t key);
    // copying -> landed, when gen is current (the boundary saw its event complete); false otherwise
    bool land(int32_t i);
    // landed -> none (published): the slot and key returned; false when not landed
    bool publish(int32_t i, int32_t & slot, int32_t & key);
    // copying or landed -> empty, a new generation (the slot stays reserved, its bytes garbage)
    bool invalidate(int32_t i);
    // every record -> none; the slots returned to the caller (reset to the boot placement)
    std::vector<int32_t> discard_all();
    // a loan returned the slots it borrowed: every reserved record's bytes are garbage (a new generation each)
    void return_from_loan();
    uint32_t gen() const { return gen_; }

  private:
    rec r_[max_recs];
    uint32_t gen_ = 1;
};

// the residents to retire for a layer's need: least decayed usage first, then key, then slot (deterministic)
std::vector<cand> victims(std::vector<cand> c, int32_t need);
// the misses to admit into n_avail reserved slots: most rows first, then route order (stable)
std::vector<miss> admit(std::vector<miss> m, int32_t n_avail);

} // namespace fn::runtime::adm
