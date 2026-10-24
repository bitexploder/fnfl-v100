#pragma once

// ME1: the host-backed KV tier (the 16 GB configuration; one or two devices (MEKV), either executor (ME3), the sparse
// draft). The
// target's BF16 K and V of every QSA layer and the draft's F16 K and V keep their whole history, at their logical
// cells, in mapped pinned host memory: the "target KV (K0)" and "draft KV (F16)" classes are allocated there
// (run_load). The device holds the "KV tier (ME1)" class: per card a K arena and a V arena of the 13 attention
// layers' windows (the 12 QSA layers in order, then the draft) of W cells each, then a gather area of
// fn::qsa::tier::gather_cells cells, and the remap's tables. Summaries, GDN state and the rest stay on the device.
//
// ME6: the raw index keys (BF16 [cell][128]) follow the same tier. The "target index keys (ME6)" (the owner's) and
// "draft index keys (ME6)" (both cards) classes hold their whole history in mapped pinned host memory; the device's
// "index key windows (ME6)" class holds a key arena a card of key_slots windows of W cells (slot a / 2 for the target
// layers the card owns, a % 2 == card; slot key_draft for the draft), sharing K/V's lo, moves and validity. Only
// summary_update reads keys, and only a completing block's cells (the newest end of the slot), so nothing is
// gathered: the window always covers them (lo <= 4 floor(first / 4)).
//
// Decode (decode, then the launches' k, v and select): the window of every layer holds cells lo .. lo + W - 1, and an
// attention launch gets k_cache = window - lo (so its new cells depth + t and its tail land in the window) and the body
// from select: the blocks below lo that any row selected, gathered from host memory into the gather area, at cells
// lo + (13 - a) W + 4 s (above every window cell, so a launch's first pass defers them like cells at or past depth;
// written before the launch). select is a remap (one CTA) and a gather (zero-copy reads over PCIe) a card a layer.
// Validity: host memory holds the latest contents of cells below lo, the window those from lo on. decode() moves the
// window only between rounds: when the round's first written cell is below lo, its last above lo + W, or the tier
// left a prompt: the window's written cells are copied to host memory, then host cells lo' .. valid - 1 into the
// window, lo' = max(0, end - W / 2) rounded down to a multiple of 4 (and at most the first written cell). Rejected
// cells and a shorter slot (a cut, a checkpoint restore, a divergent prefix) keep their meaning: contents are valid by
// length, as in a resident cache, and every cell a round reads was last written by the slot's history.
//
// Prompt (prompt, stage, write_back): entering a prompt copies the window to host memory, which then holds every cell.
// Per chunk and QSA layer a card's arenas (from cell 0, at least the slot's cells) take that layer's history below
// the chunk (stage, before the first half's K/V write); the prompt's K/V writes and attention use them; each half's
// new cells go back to host memory (write_back). The draft's prompt writes its K and V straight into host memory. The
// decode window is rebuilt by the first decode round after the prompt (inside the prefill-to-ready boundary).
// ME6: keys need no whole-slot staging: per chunk and layer the cells of the chunk's first block below its start (at
// most 3) are staged at the key arena's start (stage_keys), the chunk's key writes and pooling use the arena from
// there, and each half's new cells go back to host memory (write_back_keys).
//
// ME3, the graph executor: a captured launch cannot take a window-dependent pointer, so in graph mode (set_graph) k and
// v give each layer's window base and shift the device word holding lo (written by decode on the stream before the
// round, read by the attention launches' host-K/V instantiations: cell c at base + (c - lo) * 256, the same cells as
// eager's base - lo); select always runs the remap with lo and base's lo / 4 read from that word (at lo 0 the body is
// copied unchanged and nothing gathered: the same body eager passes through). The window, its rebuilds, the gather
// and the prompt's staging and write-back are unchanged; decode stays at the round boundary, before the launch.
// ME6: keys likewise give the key window's base, and summary_update's round form reads the same word (its shift).
//
// Copies use each shard's device and bound compute stream (MEKV), in order with its launches.

#include <cstdint>
#include <string>

namespace fn::model {

class model;

class kv_tier {
  public:
    static constexpr int32_t n_attn = 13;           // the 12 QSA layers (layer / 4), then the draft
    static constexpr int32_t draft_attn = n_attn - 1;
    static constexpr int32_t window_cells = 32768;   // the serving window W
    // ME6: the key arena's windows a card (6 owned target layers, then the draft) and the draft's slot
    static constexpr int32_t key_slots = 7;
    static constexpr int32_t key_draft = key_slots - 1;
    static constexpr int32_t prompt_key_cells = 8192 + 4; // a prompt chunk (prefill_stage_rows) and its first block

    // the arenas' cells a card: every window and the gather area, and the whole slot for a prompt's staging
    static int64_t arena_cells(int32_t window, int64_t n_cells);
    static int64_t table_bytes(); // the remap's tables a card
    static int64_t key_arena_cells(int32_t window); // ME6: the key windows a card, or one prompt chunk if larger
    // ME6: card c's key slot of attention layer a; -1 for a target layer the card does not own
    static int32_t key_slot(int32_t c, int32_t a);

    struct counters {
        int64_t gathered_blocks[2] = {0, 0}; // per card, read from the device by read_counters
        int64_t rebuilds = 0, flushes = 0;
        int64_t flush_bytes = 0, load_bytes = 0, stage_bytes = 0, write_back_bytes = 0;
        int64_t key_flush_bytes = 0, key_load_bytes = 0, key_stage_bytes = 0, key_write_back_bytes = 0; // ME6
    };

    // from m's classes (host history and the device class); window: W (window_cells when serving; kv-tier-test smaller)
    bool init(const model & m, int32_t window, std::string & err, void * const * streams = nullptr);

    // before a decode round (target or draft): its writes start at cell first, its reads and writes end below end, and
    // the slot's contents are meaningful below valid (the larger of the target's and the draft's lengths)
    bool decode(int64_t first, int64_t end, int64_t valid, void * stream, std::string & err);
    // card c's attention layer a's launch pointers for the current window (decode). ME3, graph mode (set_graph): the
    // layer's window base, with shift(c) the device word holding the window's first cell (written by decode before
    // the round), so a captured launch serves every window; eager: base - lo, shift null
    uint16_t * k(int32_t c, int32_t a) const;
    uint16_t * v(int32_t c, int32_t a) const;
    const int64_t * shift(int32_t c) const { return graph_ ? cards_[c].lo_word : nullptr; }
    // ME6: card c's layer a's index keys for the current window (decode), as k and v (graph mode: the window's base,
    // with shift(c)); a must be a key slot of c
    uint16_t * keys(int32_t c, int32_t a) const;
    // ME3: the executor the launches are enqueued for (target::set_executor); graph mode's select always remaps (the
    // remap reads lo from the device word; at lo 0 it copies the body unchanged)
    void set_graph(bool on) { graph_ = on; }
    bool graph() const { return graph_; }
    // card c's layer a: the selection of rows (body, offsets on the device) remapped and its blocks below the window
    // gathered; out: the body the launch reads (draft: kept for the chain's later steps, draft_body)
    bool select(int32_t c, int32_t a, int32_t rows, const uint32_t * body, const int32_t * offsets, uint32_t * err_word,
                void * stream, const uint32_t *& out, std::string & err);
    const uint32_t * draft_body(int32_t c) const { return draft_out_[c]; }

    // a prompt chunk starts (the window copied to host memory, once a prompt)
    bool prompt(void * stream, std::string & err);
    // card c's staging: layer a's host cells 0 .. cells - 1 into the arenas, which then read as a whole-slot cache
    bool stage(int32_t c, int32_t a, int64_t cells, void * stream, std::string & err);
    uint16_t * stage_k(int32_t c) const { return cards_[c].k_arena; }
    uint16_t * stage_v(int32_t c) const { return cards_[c].v_arena; }
    // card c's layer a: staged cells first .. first + n - 1 to host memory
    bool write_back(int32_t c, int32_t a, int64_t first, int64_t n, void * stream, std::string & err);
    // ME6: a prompt chunk's keys on card c's layer a, cells start .. start + rows - 1: host cells 4 floor(start / 4) ..
    // start - 1 into the key arena's start (once a chunk and layer, before its first key write); stage_keys_ptr then
    // reads as a key cache of the cells from 4 floor(start / 4) through the chunk (the last staging's)
    bool stage_keys(int32_t c, int32_t a, int64_t start, int64_t rows, void * stream, std::string & err);
    uint16_t * stage_keys_ptr(int32_t c) const;
    // ME6: the staged keys of cells first .. first + n - 1 (inside the last staging's chunk) to host memory
    bool write_back_keys(int32_t c, int32_t a, int64_t first, int64_t n, void * stream, std::string & err);

    // after the caller zeroed the host history (a reset): the window holds nothing, host memory every cell
    void zeroed();
    // host memory made whole (the window's written cells copied, the window kept), for state checks; synchronous
    bool flush(void * stream, std::string & err);
    bool read_counters(std::string & err);
    bool reset_counters(std::string & err); // synchronous
    std::string counters_text() const;      // after read_counters
    const counters & count() const { return n_; }
    int64_t lo() const { return lo_; }
    bool in_decode() const { return decode_; }
    // card c's layer a's host history (tests)
    uint16_t * host_k(int32_t c, int32_t a) const { return cards_[c].host_k[a]; }
    uint16_t * host_v(int32_t c, int32_t a) const { return cards_[c].host_v[a]; }
    uint16_t * host_keys(int32_t c, int32_t a) const; // ME6: null when c does not own a's keys

  private:
    struct card {
        uint16_t * host_k[n_attn] = {};
        uint16_t * host_v[n_attn] = {};
        uint16_t * k_arena = nullptr; // [arena cells][256]
        uint16_t * v_arena = nullptr;
        int32_t * table = nullptr;    // [max_blocks]
        int32_t * list = nullptr;     // [max_gather]
        int32_t * count = nullptr;    // [1]
        unsigned long long * total = nullptr; // [1]
        uint32_t * body[2] = {};      // the target's remapped body, the draft's
        int64_t * lo_word = nullptr;  // ME3: [1] the window's first cell, on the device (graph mode)
        uint16_t * host_keys[key_slots] = {}; // ME6: [cells][128] each key slot's history
        uint16_t * key_arena = nullptr;       // ME6: [key_arena_cells][128]
        int64_t key_s0 = -1;                  // ME6: the cell at the key arena's start while a prompt stages keys
        int64_t key_end = 0;                  // ME6: the end of the chunk staged
        int32_t key_a = -1;                   // ME6: the layer last staged
    };
    bool copy_window(bool to_host, int64_t first, int64_t last, void * stream, std::string & err);

    int32_t device_[2] = {-1, -1};
    void * streams_[2] = {nullptr, nullptr}; // bound compute streams; null uses the call stream (single-device tests)
    card cards_[2];
    const uint32_t * draft_out_[2] = {nullptr, nullptr}; // the draft step 0's body (select), for steps 1 ..
    int32_t w_ = 0;
    int64_t n_cells_ = 0;
    int64_t lo_ = 0, hw_ = 0; // the window's first cell, and the end of the cells written or loaded since its build
    bool decode_ = false;     // false: host memory holds every cell (after a reset or a prompt)
    bool graph_ = false;      // ME3: set_graph
    counters n_;
};

} // namespace fn::model
