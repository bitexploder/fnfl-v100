#pragma once

// PB1: the bulk prompt path on both cards (FC-PF1, FC-PF3; fnfl/design/2026-10-14-prefill-plan.md, "Fix the execution
// shape first" .. "Publication, state and handoff"). A chunk (at most 3,072 rows, halves of at most 1,536, PF1's
// partition) runs sublayer-major: for each layer, the attention sublayer of half A, then of half B, then the FFN
// sublayer of A (after A's attention exchange), then of B; the next layer's A waits only for A's FFN exchange. Every
// sublayer ends with its full-width FP16 partial copied to the peer by the copy engine (fn/xchg/prompt_xchg.h); the
// other half computes meanwhile. Target HC is replicated (FC-PF3): each card holds both halves of every coefficient
// (its resident half and the peer's replica) and the full streams S [rows][4][2560] F32. After layer 47 the draft's
// catch-up runs over the chunk's positions, half by half (the eh_proj product, the split Q8_0 HC with its two
// messages, K and V into the draft cache). The state objects keep decode's layout and SR1's versions: half A writes
// logical version 1, half B version 2 (ME11: the GDN matrices as banks: half A from the committed bank and records into
// the other bank, half B in place there); the caller publishes the chunk (fn::round::advance by the chunk's halves,
// the GDN selector by gdn_after_prompt)
// after both cards' work joined clean, so a failed chunk publishes nothing (design note, section 7).
//
// Host code only. Every launch is enqueued directly (no graph); the host computes each launch's rows and positions
// from the chunk, so nothing is allocated, reserved or synchronized between layers. Both cards are enqueued phase by
// phase (card 0, then card 1), every event recorded before the wait that reads it. The message protocol (four steps,
// design note section 6): producer -> its event -> the copy (after the peer freed its receive slot) -> its event -> the
// consumer, which checks the generation in the header; the producer's next write waits for the copy.
//
// Buffers: R1's "prefill arena (PB1)" and "HC and PLE replicas (PB1)" classes (fn/model/state.h), carved here.

#include "fn/model/draft.h"
#include "fn/model/ladder.h"
#include "fn/model/model.h"
#include "fn/model/pass.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fn::model {

constexpr int32_t prompt_chunk_rows = 8192; // PB3: a whole 8,192 span (was 3,072 in halves of 1,536)
constexpr int32_t prompt_half_rows = 4096;
// MUP2 piece 1: a chunk's sublayers run as pieces in row order (today's halves generalized), at most this many a chunk
// (the P messages' generation carries the piece in 7 bits, fn/xchg/prompt_xchg.h pgen): 128, so the correctness cases'
// 64-row pieces cover an 8,192-row chunk; serving's pieces are at least 1,024 rows (prefill.h pf_piece_floor_rows), at
// most 8 a chunk
constexpr int32_t prompt_max_pieces = 128;
constexpr int32_t prompt_piece_grid = 64;  // every cut sits on the 64-row grid from the chunk's start (a GDN block)

// the arena's bytes a card (n_cells sizes nothing here: the QSA slab is the full 262,144-cell scratch, 32 MiB); ME3:
// for a layout of chunks of at most chunk_rows rows in halves of at most half_rows (prompt_path::layout); MX6: single
// omits peer receive buffers and card 1's shared buffers (aliased to card 0 in set_links)
int64_t prompt_arena_bytes(int32_t chunk_rows = prompt_chunk_rows, int32_t half_rows = prompt_half_rows,
                           bool single = false, int32_t card = 0);
// the replicas' bytes a card: the peer's half of the 96 target HC sublayers' coefficients and of PLE's, and the draft's
// full enorm and hnorm; single (ME1, both logical cards on one device): the draft's full norms only, the peer's HC
// and PLE tensors being read where they are resident
// TTTS3: the PLE key and value's replicas in their format (ple_kv: UD-Q4_K_XL's Q8_0), the taps F32 with ple_conv32
// (TTTS2: twice the F16 bytes)
int64_t prompt_replica_bytes(bool            single     = false,
                             fn::ple::kv_fmt ple_kv     = fn::ple::kv_fmt::bf16,
                             bool            ple_conv32 = false);

// the declared bulk call sites (design note section 4): host counters, one an enqueue
enum class psite : int32_t {
    embed,
    hc_attn,
    hc_ffn,
    gdn_input,
    gdn_core,
    gdn_out,
    qsa_input,
    qsa_index,
    qsa_attend,
    qsa_out,
    ffn_router,
    ffn_experts,
    ffn_shared,
    ple,
    exchange,
    draft,
    head,
    count
};
constexpr int32_t n_psites = static_cast<int32_t>(psite::count);
const char *      psite_name(psite s);

// APF2, the kernel-size ladder: each product's route from the measured table (fn/model/ladder.h: the entry at the
// largest rung at or below the piece's rows; routed rows an expert for the grouped experts), else its retained route:
//   dense:  a ladder tile (fn::dense::pl_tc: the packed weights decoded in the tile loader, split K), or pd_convert at
//           the sublayer's piece 0 (when any piece of the chunk takes the retained route) then pd_tc
//   hc:     both products on ladder tiles (fn::hc::hct_mix_ladder), or hct_convert then hct_mix
//   expert: gate/up and down each grouped (fn::experts ep_*_ladder, resident Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down;
//           IQ formats and the expert service's staged or ring layers keep per-task), or per-task
//   graph:  each unit (chunk_piece) one CUDA graph, captured once a key and replayed with its kernels' parameters
//           updated (no stream capture), at most prompt_graph_rows rows, on the two-card path with no expert
//           service, no host-backed KV tier and no single-device pair; prompt_graphs(false) turns it off (the server's
//           props switch)
// Split K sums in another order, so routes agree within the families' FP64 bounds and section 5's KL line, not
// bitwise; a route reruns bitwise.
//   rung:   the smallest of 64, 128, .. 8,192 at least the chunk's rows (the graph cut-over; reported)
constexpr int32_t prompt_graph_rows = 512;

// the graph switch (process-wide; default on)
void prompt_graphs(bool on);
bool prompt_graphs_on();

struct prompt_route {
    int32_t rung  = 0;
    bool    graph = false;
};

// APF2: the ladder keys of the prompt path's products (the call sites' and prompt-tune's): a dense product of
// fn::dense format f in layout lay (fn::dense::pd_layout), K k, N n; an HC product; an expert product of
// fn::experts format f; the routed rows an expert of a piece of rows; the cost model of key on device
fn::model::lad_key prompt_dense_key(int32_t f, int32_t lay, int32_t k, int32_t n);
fn::model::lad_key prompt_hc_key(bool up, bool q8);
fn::model::lad_key prompt_expert_key(bool down, int32_t f);
int32_t            prompt_expert_rows(int32_t rows);
bool               prompt_lad_model(const fn::model::lad_key & key, int32_t device, fn::model::lad_model & m);
// the distinct ladder keys of card's products (the window's layers; the draft's with d), in first-use order
std::vector<fn::model::lad_key> prompt_ladder_keys(const pass_card &  p,
                                                   const draft_card * d,
                                                   const window &     win,
                                                   int32_t            card);

// what the path allows: graph_ok the two-card, no-service, no-tier condition. force: TEST ONLY, each field 0 the
// table's choice, 1 the new routes (each product the cost model's first-ranked ladder configuration; the grouped
// experts wherever the format takes them; graphs on), 2 the retained routes (graphs off)
struct prompt_route_force {
    int32_t dense = 0, hc = 0, expert = 0, graph = 0;
};

prompt_route prompt_route_for(int32_t rows, bool graph_ok, const prompt_route_force & force = {});
// "rung 512, graph" (the read logs' route field)
std::string  prompt_route_name(const prompt_route & r);

// one chunk as the driver hands it over
struct prompt_chunk {
    int64_t         index   = 0;  // the run's chunk number (the HC dump's chunk selection, the token slot's parity)
    int64_t         seq     = 0;  // the prompt path's chunk number since init, never repeated (message generations)
    int64_t         start   = 0;  // its first position
    int32_t         rows    = 0;
    int32_t         half[2] = { 0, 0 };
    const int32_t * tok     = nullptr;               // host: the chunk's tokens
    const float *   e[2]    = { nullptr, nullptr };  // device: each card's PLE input rows [rows][2560] (PF1's staging)
    bool            draft   = true;
    bool            more    = false;  // ME4: another chunk of the run follows (expert_service::prompt_layer)
    // MUP1 piece 5, the prompt-row interface (VIS1 fills the supplied forms; fn/kernels/qsa/imrope.cuh). A row is a
    // token id (tok) or a supplied embedding, with four rope planes (t, h, w, e), a PLE id and its owner (the slot the
    // path is bound to). rope_off: the slot's rotary offset; a text row's planes are all its cell plus it (0 today,
    // its bits unchanged). planes[c]: device rows on card c, [rows][4] (null: text). A row's PLE id is its tok entry
    // (text: its own token; an image row: the image token, fn/runtime/image_rows.h)
    int32_t         rope_off  = 0;
    const int32_t * planes[2] = { nullptr, nullptr };
    // VIS1 Stage B: with planes, the QSA blocks the chunk completes, start / 4 .. (start + rows) / 4 - 1, each its
    // first row's planes ([blocks][4] on card c; a block's first row can precede the chunk): the target's and the
    // draft's index summaries are roped at them (fn/kernels/qsa/qsa.h update_args::planes). Null with planes null
    const int32_t * bplanes[2] = { nullptr, nullptr };

    // VIS1 Stage B: supplied embedding rows (images), runs of chunk rows that take supplied rows in place of their
    // token's embedding (fn/kernels/embed embed_supply). embd[c]: card c's half of the runs' rows, packed in run order,
    // [sum of n][1280] F32 on card c's device (single: both halves on card 0's). Runs ascending and disjoint (one
    // embed_supply launch a run and card); none: text
    struct supply_run {
        int32_t row0 = 0, n = 0;
    };

    const supply_run * runs                     = nullptr;  // host
    int32_t            n_runs                   = 0;
    const float *      embd[2]                  = { nullptr, nullptr };
    // VIS1 Stage B: with runs, the image token's embedding row [2560] F32 on card c (vision_service::token_row), the
    // draft's input at the runs' rows. The reference's draft reads no image rows; FNFL's catches up over them with the
    // image token's embedding, as the split's back (which embeds the tokens itself) does. Null where nothing drafts
    const float *      draft_e[2]               = { nullptr, nullptr };
    // MUP2 piece 1: the chunk's pieces, rows each in row order, fixed for the chunk (prompt_set_pieces); n_pieces 0:
    // the halves (one piece a half). A partition refines the halves (a cut at half[0]) and cuts on the 64-row grid,
    // so every family's work over a piece is bitwise its share of the whole sublayer and the draft's catch-up stays on
    // the halves
    int32_t         n_pieces  = 0;
    int32_t         piece[prompt_max_pieces] = {};
};

// MUP2 piece 1: c's pieces (the halves when n_pieces is 0), piece j's first row and rows, and the logical state
// versions piece j reads and writes: piece 0 reads the committed version 0 (ME11's records replayed there) and writes
// 1; each later piece reads its predecessor's and writes the other of 1 and 2 (the halves' 0 -> 1 -> 2), so the chunk
// commits the last piece's version (target::prompt_commit's k)
int32_t prompt_pieces(const prompt_chunk & c);
int32_t prompt_piece_r0(const prompt_chunk & c, int32_t j);
int32_t prompt_piece_rows(const prompt_chunk & c, int32_t j);

inline int32_t prompt_piece_v_in(int32_t j) {
    return j == 0 ? 0 : (j - 1) % 2 + 1;
}

inline int32_t prompt_piece_v_out(int32_t j) {
    return j % 2 + 1;
}

inline int32_t prompt_last_version(const prompt_chunk & c) {
    return prompt_piece_v_out(prompt_pieces(c) - 1);
}

// the halves cut into pieces of at most rows rows (a multiple of the grid; 0: the halves themselves), each half in the
// fewest such pieces, evened out on the grid (the last the rest). PRD1 item 12: min_rows > 0, no piece made by
// subdividing a half below min_rows (fewer pieces, possibly above rows, or a larger last piece; a half that cannot
// split so stays whole). false and err: a size off the grid, or more than prompt_max_pieces pieces
bool prompt_set_pieces(prompt_chunk & c, int32_t rows, std::string & err, int32_t min_rows = 0);
// c's partition is valid: its rows sum, the cut at half[0], every cut on the grid, at most max pieces
bool prompt_pieces_ok(const prompt_chunk & c);

// what the HC dump (PB2's inputs) takes from one chunk: layers, and every stride-th row of each half
struct hc_dump_spec {
    std::vector<int32_t> layers;
    std::vector<int64_t> chunks;          // chunk indices of the run
    int32_t              row_stride = 0;  // 0: no dump
    std::string          dir;
};

class prompt_path {
  public:
    prompt_path();
    ~prompt_path();
    prompt_path(const prompt_path &)             = delete;
    prompt_path & operator=(const prompt_path &) = delete;

    // the arenas, replicas (filled now, card to card), copy streams and events; p[c] is the target's pass on card c
    // (its stream is the compute stream), d[c] its draft (null: no draft). Synchronous. MU1: arena_class names the
    // arena (the other-reader lane's, "other-reader arena (MU1)"), laid out for chunks of at most rc_max rows in
    // halves of at most rh_max (the largest layout this path takes)
    bool init(model & m, pass_card * const p[2], draft_card * const d[2], std::string & err,
              const char * arena_class = "prefill arena (PB1)", int32_t rc_max = prompt_chunk_rows,
              int32_t rh_max = prompt_half_rows);
    // MU1: the slot whose chunks run next: its passes and drafts (a slot's frame; the replicas and weights are every
    // slot's). Between chunks only (no chunk of this path in flight); the next chunk's run_start reads its state.
    void bind(pass_card * const p[2], draft_card * const d[2]);

    // ME2b: the arena (and PF1's staging) borrowed from the expert pool (on), or returned (off), through the cards'
    // expert service (layer.h expert_service::prompt_hold; nothing without one, or with nothing lent). A borrow
    // clears the error words (the region held expert slots since the last prompt). Synchronous. ME3: need, the bytes
    // the prompt writes (the arena's layout, arena_spans, and the caller's staging); null: the whole region
    bool hold(bool on, std::string & err, const std::vector<byte_span> * need = nullptr);

    // ME3: the arena's views carved for chunks of at most rc rows in halves of at most rh (from its base, so the
    // layout's writes are a prefix of the arena: arena_spans); not under a sized loan. Host only. init's is the full
    // 8,192 / 4,096
    bool layout(int32_t rc, int32_t rh, std::string & err);
    void arena_spans(std::vector<byte_span> & out) const;
    // ME3: the loan's startup check through the cards' expert service (layer.h prompt_loan_check)
    bool loan_check(const std::vector<byte_span> & need, std::string & err) const;
    expert_service * service() const;
    int32_t layout_rows() const { return rc_; }
    int32_t layout_half() const { return rh_; }

    // a run's start, both cards drained: the predecessor row (the draft's retained carry, both halves) and PLE's full
    // history from the committed version (cur: the cards' committed slot). Synchronous.
    bool run_start(uint32_t cur, std::string & err);

    // enqueue one chunk on both cards (no wait); the caller joins both compute streams, then checks errors
    bool chunk(const prompt_chunk & c, std::string & err);
    // MU1: chunk's three parts, in order, for a chunk run in layer groups: its tokens and the embedding (begin), the
    // layers lo .. hi - 1 (layers; groups in order, covering 0 .. 47 once), then layer 47's last exchange and the
    // draft's catch-up (end). The kernels and their order are chunk's; between two parts the host may wait for the
    // cards and run other slots' rounds, which touch none of this path's arena, links or the bound slot's state.
    bool chunk_begin(const prompt_chunk & c, std::string & err);
    bool chunk_layers(const prompt_chunk & c, int32_t lo, int32_t hi, std::string & err);
    bool chunk_end(const prompt_chunk & c, std::string & err);
    // MUP2 piece 1: one unit of chunk_layers' loop, sublayer sub of layer l for piece j (its four phases): the units
    // in order (layer, sublayer, piece) are chunk_layers' kernels in its order, so the host may pause between any two
    // of them (the lane keeps every operand). A sublayer's weights are converted at its piece 0 and read by its other
    // pieces, so its pieces run before the next sublayer's
    bool   chunk_piece(const prompt_chunk & c, int32_t l, int32_t sub, int32_t j, std::string & err);
    // MU1: this path's copy stream on card c (a layer group's completion waits for it as for the compute streams)
    void * copy_stream(int32_t c) const { return copy_[c]; }

    // after a clean join: both cards' prompt error words and the selection's and EP1's words; cleared after reading.
    // A set word fails the chunk (the caller publishes nothing).
    bool check_errors(std::string & err);

    // after a clean join and the publication: the retained carry (the chunk's last row's own halves) and the
    // predecessor row for the next chunk; enqueued on the compute streams
    bool publish(const prompt_chunk & c, std::string & err);

    // the last row's own half of the final streams into the pass's layer-47 boundary row 0 (hpost = hx = 0) for the
    // decode head (the caller runs it)
    bool head_input(const prompt_chunk & c, std::string & err);

    // TTTS1: a split stage's crossing of chunk c at layer k, the back's first layer. Per half, the state layer k's first
    // sublayer takes (PB3 folds layer k - 1's FFN exchange into layer k's HC, so the exchange is not combined here):
    // the half's streams S (card 0's copy), each card's inject rows, and each card's FFN partial message (its header,
    // with its generation, and its FP16 payload), laid out for c as crossing_bytes counts them. crossing_out (the
    // front, after chunk_layers up to k) enqueues the copies device to host on the compute streams and releases the FFN
    // links as their consumer would; crossing_in (the back, after chunk_begin) enqueues them host to device and stands
    // for the exchange's copies (the links' generations and events), so layer k runs as on one box. The back's chunk
    // carries the front's seq (the generations): the caller sets it.
    static int64_t crossing_bytes(const prompt_chunk & c);
    bool           crossing_out(const prompt_chunk & c, int32_t k, uint8_t * dst, std::string & err);
    bool           crossing_in(const prompt_chunk & c, int32_t k, const uint8_t * src, std::string & err);

    // TEST ONLY: card r's prompt error word set during the next chunk (a failed chunk's invalidation)
    void test_fail_next(int32_t card) { test_fail_ = card; }

    // TEST ONLY (prefill-test dry, PB1's determinism follow-up): every arena byte set to 0xFF (a NaN in F32, FP16
    // and BF16) and the error words cleared, so a run that reads arena data an earlier run left shows it even on zero
    // weights. Synchronous. ME2b: borrows a lent arena first and keeps it (the next prefill run returns it).
    bool test_poison(std::string & err);

    // TEST ONLY: the device final streams of chunk row r on card c (the draft check's hidden rows)
    const float * streams_row(int32_t card, int32_t r) const;
    const float * pred_row(int32_t card) const;

    // TEST ONLY (prefill-test vcheck, the PB1 review's V check): when set, each draft half first overwrites its rows of
    // the final streams with rows [pos0, pos0 + n) of host (pinned, [n_rows][4][2560] F32 by absolute position), in
    // stream order, so the bulk draft span reads given hidden rows; null: none
    void test_hidden(const float * host, int64_t n_rows) {
        test_h_      = host;
        test_h_rows_ = n_rows;
    }

    // TEST ONLY (SP2 Amendment 1, vcheck's sparse form): when set, each card's draft half copies its FP16 mixed rows
    // (the K, V and index-key products' operand) into rows [pos0, pos0 + n) of host[card] (pinned, [n_rows][2560]
    // FP16 by absolute position); null: none
    void test_mixed(uint16_t * host0, uint16_t * host1, int64_t n_rows) {
        test_m_[0]   = host0;
        test_m_[1]   = host1;
        test_m_rows_ = n_rows;
    }

    // TEST ONLY (SP2 Amendment 1 item 7, vcheck's localization; with test_mixed's rows): when set, each card's draft
    // half also copies its eh_proj output (the split HC's streams, [n_rows][4][1280] F32, this card's features) and
    // its HC low ([n_rows][320] F32) by absolute position; null: none
    void test_stages(float * eh0, float * eh1, float * low0, float * low1) {
        test_eh_[0]  = eh0;
        test_eh_[1]  = eh1;
        test_low_[0] = low0;
        test_low_[1] = low1;
    }

    void set_hc_dump(const hc_dump_spec & s) { dump_ = s; }

    const std::array<int64_t, n_psites> & site_counts() const { return sites_; }

    void clear_site_counts() { sites_.fill(0); }

    // APF1 Part 4: the current chunk's routes (chosen at chunk_begin) and the executed variants since the last clear:
    // host counts of the launches each route made (test-only evidence; nothing reads them in serving)
    const prompt_route & route() const { return route_; }

    struct route_counts {
        int64_t  dense_ladder = 0, dense_retained = 0;   // dense products (pl_tc calls; pd_tc calls)
        int64_t  dense_split = 0;                        // of dense_ladder: split K
        int64_t  hc_ladder = 0, hc_retained = 0;         // HC sublayer pieces
        int64_t  ex_grouped = 0, ex_per_task = 0;        // gate/up launches: grouped, per-task
        int64_t  exd_grouped = 0, exd_per_task = 0;      // down launches
        int64_t  graph_units = 0, eager_units = 0;       // units replayed as a graph, run eagerly
        int64_t  graph_captures = 0, graph_updated = 0;  // graph captures; replays by parameter update
        int64_t  qsa = 0, gdn = 0, ple = 0;              // the retained QSA attention, GDN core and PLE pieces
        uint32_t dense_fmts = 0;                         // bits fn::dense::fmt of the ladder products (bf16: BF16)
        uint32_t ex_fmts    = 0;                         // bits fn::experts::fmt of the routed gate/up launches

        // this += a - b (the counts), the format bits a's added
        void        add_since(const route_counts & a, const route_counts & b);
        // "dense ladder 79 (Q8_0; split 12) / retained 0, hc ladder 24 / retained 0, ..." (logs and coverage)
        std::string text() const;
    };

    const route_counts & route_count() const { return rcount_; }

    void clear_route_counts() { rcount_ = route_counts{}; }

    // TEST ONLY: forced routes (prompt_route_force) for the following chunks
    void test_force_route(const prompt_route_force & f) { force_ = f; }

    // TEST ONLY (the coverage cases' synthetic expert assignment): with on, each card's routed experts of chunk row r
    // are replaced after the router by slot s: s < 4 ? s : 4 + (37 r + 61 s) % 400 (four experts take every row, 108
    // take none), so the grouped route is reached predictably; the routing weights are the router's
    bool test_skew_experts(bool on, std::string & err);

    // the device bytes of the cached graph executables (an estimate: their node count times the kernel parameter
    // bound); the memory account's figure
    int64_t graph_cache_nodes() const { return graph_nodes_; }

  private:
    struct card_arena;
    struct link;
    bool carve(card_arena & a, int32_t rc, int32_t rh, bool single, int32_t card, std::string & err);
    void set_links();
    bool fill_replicas(std::string & err);
    bool      prepare(std::string & err);  // APF1 Part 4: the prepared scales (fill_replicas' replicas read)
    bool      unit_graph(const prompt_chunk & c, int32_t l, int32_t sub, int32_t j, std::string & err);
    bool      unit_eager(const prompt_chunk & c, int32_t l, int32_t sub, int32_t j, std::string & err);
    // under a unit's capture: a wait on an event recorded before the capture is satisfied by the pre-join (skipped);
    // an event recorded during it is re-recorded after the replay. Outside a capture: the plain calls
    bool      ev_wait(void * stream, void * event, std::string & err, const char * what);
    bool      ev_record(void * event, int32_t card, void * stream, std::string & err, const char * what);
    bool      peer_copy(void *        dst,
                        int32_t       dst_card,
                        const void *  src,
                        int32_t       src_card,
                        size_t        bytes,
                        void *        stream,
                        std::string & err);
    bool layer_half(const prompt_chunk & c, int32_t layer, int32_t sub, int32_t h, int32_t phase, std::string & err);
    bool draft_half(const prompt_chunk & c, int32_t h, int32_t phase, std::string & err);
    bool dump_hc(const prompt_chunk & c, int32_t layer, int32_t sub, int32_t h, std::string & err);
    // MUP2: piece j's P message in card's send (or receive) region: the pieces' messages lie in row order, each a
    // header and its rows' payload
    uint8_t * p_msg(int32_t card, bool send, const prompt_chunk & c, int32_t j) const;

    void count(psite s) { ++sites_[static_cast<size_t>(s)]; }

    model *                       m_       = nullptr;
    bool                          lad_on_  = false;  // APF2: the table applies (dense_q8_of, once at init)
    pass_card *                   p_[2]    = { nullptr, nullptr };
    draft_card *                  d_[2]    = { nullptr, nullptr };
    void *                        copy_[2] = { nullptr, nullptr };
    std::vector<card_arena>       a_;
    std::vector<link>             links_;
    std::array<int64_t, n_psites> sites_{};
    int32_t                       hist_cur_  = 0;  // which of the two PLE full histories holds the latest
    int32_t                       rc_ = prompt_chunk_rows, rh_ = prompt_half_rows;  // ME3: the layout
    int32_t rc_max_ = prompt_chunk_rows, rh_max_ = prompt_half_rows;                // MU1: the arena's largest layout
    bool                          held_ = false, held_whole_ = false;  // ME3: borrowed (hold), the whole region
    int32_t                       test_fail_ = -1;
    const float *                 test_h_      = nullptr;
    int64_t                       test_h_rows_ = 0;
    uint16_t *                    test_m_[2]   = { nullptr, nullptr };
    int64_t                       test_m_rows_ = 0;
    float *                       test_eh_[2]  = { nullptr, nullptr };
    float *                       test_low_[2] = { nullptr, nullptr };
    hc_dump_spec                  dump_;
    // APF1 Part 4: the ladder
    prompt_route                  route_;
    prompt_route_force            force_;
    route_counts                  rcount_;
    bool                          capturing_ = false;
    std::vector<void *>           cap_rec_[2];             // events recorded during the capture, per card
    void *                        gj_[2][4] = {};          // pre-join, fork, join and done events per card
    // APF2: the unit graphs, captured once a key (unit_key) and replayed by parameter update (graph_rec.h)
    struct unit_graph_entry;
    std::unordered_map<uint64_t, std::shared_ptr<unit_graph_entry>> ugraphs_;
    uint64_t                      unit_key(const prompt_chunk & c, int32_t l, int32_t sub, int32_t j) const;
    int64_t                       graph_nodes_ = 0;
    int32_t *                     skew_host_   = nullptr;  // pinned [RC][10] (test_skew_experts)
    bool                          skew_        = false;
    // APF2: the ladder's selection: a product's route at rows (force: prompt_route_force's field), whether any piece
    // of c takes the retained route for key (rows the pieces' times mul; experts: routed rows an expert), the cost
    // model of key on this path's card 0 (cached)
    std::vector<std::pair<fn::model::lad_key, fn::model::lad_model>> lad_models_;
    fn::model::lad_cfg            lad_route(const fn::model::lad_key & key, int32_t rows, int32_t force);
    bool                          lad_retained_any(const fn::model::lad_key & key,
                                                   const prompt_chunk &       c,
                                                   int32_t                    force,
                                                   int32_t                    mul     = 1,
                                                   bool                       experts = false);
    const fn::model::lad_model &  lad_model_of(const fn::model::lad_key & key);
    void                          note(int64_t route_counts::* f);  // rcount_ and the process's totals
    void                          note_fmt(uint32_t route_counts::* f, int32_t bit);
};

// TEST ONLY: the executed-variant counts of every prompt_path in the process (the coverage table's evidence: a case
// reads them before and after it runs), and the rungs of the chunks begun (bit log2(rung / 64)); clear_fmts: the
// format bits cleared after the read (a case's own formats from there)
void prompt_route_totals(prompt_path::route_counts & out, uint32_t & rungs, bool clear_fmts = false);
// TEST ONLY: the totals collected (off by default: serving takes no lock for them; each path's own counts, the read job
// line's, are kept either way)
void prompt_route_collect(bool on);

}  // namespace fn::model
