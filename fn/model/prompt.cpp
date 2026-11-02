#include "fn/model/prompt.h"

#include "fn/kernels/dense/dense_ladder.h"
#include "fn/kernels/dense/dense_prompt.h"
#include "fn/kernels/draft/draft_prompt.h"
#include "fn/kernels/embed/embed.h"
#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/experts_ladder.h"
#include "fn/kernels/experts/experts_prompt.h"
#include "fn/kernels/experts/experts_share.h"
#include "fn/kernels/gdn/gdn_prompt.h"
#include "fn/kernels/hc/hc.h"
#include "fn/kernels/hc/hc_prompt.h"
#include "fn/kernels/ple/ple.h"
#include "fn/kernels/ple/ple_prompt.h"
#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/kernels/qsa/qsa_prompt.h"
#include "fn/kernels/round/version.h"
#include "fn/model/graph_rec.h"
#include "fn/model/kv_tier.h"
#include "fn/model/ladder.h"
#include "fn/model/shapes.h"
#include "fn/xchg/prompt_xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace fn::model {

namespace {

constexpr int32_t D               = fn::n_embd;     // 2560
constexpr int32_t F               = fn::hc::n_own;  // 1280
constexpr int32_t RC              = prompt_chunk_rows;
constexpr int32_t RH              = prompt_half_rows;
constexpr int32_t NE              = fn::n_expert;
constexpr int32_t NU              = fn::n_expert_used;
constexpr int32_t QF              = 12 * 2 * 256;                   // q with its gate, a card
constexpr int32_t GC              = gdn_conv_channels_card;
constexpr int32_t GV              = gdn_heads_card * gdn_head_dim;  // 3072
constexpr int64_t hc_down_bytes   = int64_t(4) * fn::hc::n_rows_down * F * 2;
constexpr int64_t hc_up_bytes     = int64_t(F) * 4 * fn::hc::n_down * 2;
constexpr int64_t hc_gamma_bytes  = int64_t(4) * F * 4;
constexpr int64_t ple_key_bytes   = int64_t(4) * F * D * 2;
constexpr int64_t ple_value_bytes = int64_t(F) * D * 2;
constexpr int64_t ple_norm_bytes  = int64_t(4) * F * 4;
constexpr int64_t ple_conv_bytes  = int64_t(4) * F * 4 * 2;

// TTTS3: the key's 4 F rows and the value's F rows of D in their stored format (ple.h kv_row_bytes); the taps F16, or
// F32 (TTTS2)
constexpr int64_t ple_key_bytes_of(fn::ple::kv_fmt f) {
    return int64_t(4) * F * fn::ple::kv_row_bytes(f);
}

constexpr int64_t ple_value_bytes_of(fn::ple::kv_fmt f) {
    return int64_t(F) * fn::ple::kv_row_bytes(f);
}

constexpr int64_t ple_conv_bytes_of(bool conv32) {
    return conv32 ? 2 * ple_conv_bytes : ple_conv_bytes;
}

static_assert(ple_key_bytes_of(fn::ple::kv_fmt::bf16) == ple_key_bytes &&
                  ple_value_bytes_of(fn::ple::kv_fmt::bf16) == ple_value_bytes,
              "the BF16 rows' replicas as before");
constexpr int64_t ple_hist_card   = fn::ple::hist_version_floats;  // [5120][9]

constexpr int64_t al(int64_t n) {
    return (n + 255) / 256 * 256;
}

bool fail(std::string & err, const std::string & what) {
    err = what;
    return false;
}

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e != cudaSuccess) {
        err = std::string("prompt: ") + what + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

// VIS1 Stage B: the draft's embeddings E [rows][2560] of half rows r0 .. r0 + n - 1 take the image token's row at
// the chunk's supplied rows (embed_supply wrote the image's rows there for the target): the row once, then doubling
// copies of the rows already written. Exact copies, in stream order before the draft reads E
bool draft_image_rows(const prompt_chunk & c,
                      int32_t              k,
                      int32_t              r0,
                      int32_t              n,
                      float *              E,
                      cudaStream_t         s,
                      std::string &        err) {
    if (c.draft_e[k] == nullptr) {
        return fail(err, "prompt: supplied rows and no image-token row for the draft (vision_service::token_row)");
    }
    const size_t row = size_t(D) * 4;
    for (int32_t i = 0; i < c.n_runs; ++i) {
        const int32_t lo = std::max(c.runs[i].row0, r0), hi = std::min(c.runs[i].row0 + c.runs[i].n, r0 + n);
        float * const dst = E + int64_t(lo) * D;
        if (lo < hi && !cuda_ok(cudaMemcpyAsync(dst, c.draft_e[k], row, cudaMemcpyDeviceToDevice, s), "draft e", err)) {
            return false;
        }
        for (int32_t done = 1; lo + done < hi;) {
            const int32_t m = std::min(done, hi - lo - done);
            if (!cuda_ok(cudaMemcpyAsync(dst + int64_t(done) * D, dst, row * size_t(m), cudaMemcpyDeviceToDevice, s),
                         "draft e", err)) {
                return false;
            }
            done += m;
        }
    }
    return true;
}

// the slot sizes (header and payload); ME3: for a layout of rc chunk rows in halves of rh (prompt_path::layout).
// MUP2: the P messages of a chunk's pieces lie in one region a card and direction, in row order, each its header then
// its rows' FP16 payload: room for the most pieces' headers and both halves' rows (a partition refines the halves)
constexpr int64_t p_region(int64_t rh) {
    return int64_t(prompt_max_pieces) * fn::xchg::pmsg_payload_at + 2 * rh * D * 2;
}
constexpr int64_t emb_slot(int64_t rc) {
    return 256 + rc * F * 4;
}
constexpr int64_t sel_slot  = 256 + fn::qsa::prompt_sel_bytes;
constexpr int64_t mom_slot(int64_t rh) {
    return 256 + rh * fn::hc::hcp_mom * 4;
}
constexpr int64_t dmix_slot(int64_t rh) {
    return 256 + rh * F * 4;
}

// the family scratch's size for halves of H rows: the largest family (design note section 5); each family carves it
// by the half's own rows
int64_t hc_base_bytes(int64_t H) {
    return al(H * fn::hc::hcp_mom * 4) + al(H * 320 * 4) + al(H * 4 * 4) + al(fn::hc::hct_scratch_bytes(int32_t(H)));
}

int64_t family_bytes(int64_t H) {
    const int64_t gdn = al(H * (GC + GV) * 2) + al(H * GC * 4) + al(H * GV * 2) + 4 * al(H * 24 * 4) + al(H * GV * 2) +
                        al(fn::gdn::prompt_wy_bytes(int32_t(H)));
    const int64_t qsa = al(H * QF * 4) + 2 * al(H * 256 * 4) + al(H * 512 * 4) + al(H * 128 * 4) +
                        al(static_cast<int64_t>(fn::qsa::prompt_ws_bytes(fn::qsa::max_cells))) + al(H * GV * 2);
    const int64_t ffn = al(int64_t(H) * NE * 4) + al(int64_t(H) * 8 * 4) + 2 * al(int64_t(H) * 4) +
                        al(int64_t(H) * NU * 4) * 2 + al(int64_t(513) * 4) + 2 * al(int64_t(H) * NU * 4) +
                        al(int64_t(fn::experts::ep_max_tasks) * 16) + al(int64_t(8) * 4) + al(int64_t(513) * 4) +
                        2 * al(int64_t(fn::experts::ep_max_ltasks) * 16) +  // APF2: the grouped task lists
                        al(int64_t(H) * NU * 320 * 2) + al(H * NU * D * 2) + al(H * D * 4) + 2 * al(H * 320 * 4) +
                        al(H * 320 * 2);
    const int64_t ple = al(H * 4 * D * 4) * 2 + al(H * D * 4) + al(H * 8 * 4);
    const int64_t drf = al(H * D * 2) + al(H * 4 * D * 2) + al(H * 4 * F * 4) + al(H * 320 * 4) + 2 * al(H * 4 * 4) +
                        al(H * D * 2) + 2 * al(H * 256 * 4) + al(H * 128 * 4); // SP2: the draft's index key
    const int64_t hc = hc_base_bytes(H);
    return std::max({gdn, qsa, ffn, ple, drf, hc});
}

int64_t family_bytes_single(int64_t rc, int64_t H, int32_t card) {
    const int64_t emb = al(emb_slot(rc));
    const int64_t gdn_c0 =
        al(H * (GC + GV) * 2) + al(H * GC * 4) + al(H * GV * 2) + 4 * al(H * 24 * 4) + al(H * GV * 2);
    const int64_t gdn_c1 = al(fn::gdn::prompt_wy_bytes(int32_t(H)));
    const int64_t qsa = al(H * QF * 4) + 2 * al(H * 256 * 4) + al(H * 512 * 4) + al(H * 128 * 4) +
                        al(static_cast<int64_t>(fn::qsa::prompt_ws_bytes(fn::qsa::max_cells))) + al(H * GV * 2);
    const int64_t ffn_rt  = al(int64_t(H) * NE * 4) + al(int64_t(H) * 8 * 4) + 2 * al(int64_t(H) * 4) +
                            al(int64_t(H) * NU * 4) * 2 + al(int64_t(513) * 4) + 2 * al(int64_t(H) * NU * 4) +
                            al(int64_t(fn::experts::ep_max_tasks) * 16) + al(int64_t(8) * 4) + al(int64_t(513) * 4) +
                            2 * al(int64_t(fn::experts::ep_max_ltasks) * 16);
    const int64_t ffn_bdy = al(int64_t(H) * NU * 320 * 2) + al(H * NU * D * 2) + al(H * D * 4);
    const int64_t ffn_sh = 2 * al(H * 320 * 4) + al(H * 320 * 2);
    const int64_t ple_c0 = al(H * 4 * D * 4) + al(H * D * 4) + al(H * 8 * 4);
    const int64_t ple_c1 = al(H * 4 * D * 4);
    const int64_t drf = al(H * D * 2) + al(H * 4 * D * 2) + al(H * 4 * F * 4) + al(H * 320 * 4) + 2 * al(H * 4 * 4) +
                        al(H * D * 2) + 2 * al(H * 256 * 4) + al(H * 128 * 4);
    const int64_t hc_c0 = hc_base_bytes(H) + al(H * D * 4);
    return card == 0 ? std::max({emb, gdn_c0, qsa, ffn_rt + ffn_bdy + ffn_sh, ple_c0, drf, hc_c0})
                     : std::max({emb, gdn_c1, qsa, ffn_bdy, ple_c1, drf});
}

// PB3's conversion region: a sublayer's FP16 weights (fn::hc::hct_convert's, then pd_convert's and
// pd_convert_bf16's), converted at its first half and read by both halves. The dense part is the largest sublayer's:
//   GDN: qkv and z fused, ssm_out, alpha and beta fused (BF16, their column scales after them);
//   QSA: wq, wk and wv fused, wo, the index q and k fused (the index owner; BF16, scales);
//   FFN: the shared expert's gate and up fused, its down, the router (BF16, scales; PB3: then the shared gate's row
//        and 7 zero rows, so the router's product carries the gate logit as column 512).
// Places in FP16 elements from the dense part's start, each 256-byte aligned.
constexpr int64_t conv_gdn = int64_t(GC + GV) * D * 2;
constexpr int64_t conv_qsa = int64_t(QF + 512) * D * 2;
constexpr int64_t conv_ffn = int64_t(640) * D * 2;
constexpr int64_t gdn_out_at = al(conv_gdn) / 2;
constexpr int64_t gdn_ab_at = gdn_out_at + al(int64_t(D) * GV * 2) / 2;
constexpr int64_t gdn_ab_s_at = gdn_ab_at + al(int64_t(48) * D * 2) / 2;
constexpr int64_t gdn_end = gdn_ab_s_at + al(int64_t(48) * 4) / 2;
constexpr int64_t qsa_out_at = al(conv_qsa) / 2;
constexpr int64_t qsa_idx_at = qsa_out_at + al(int64_t(D) * GV * 2) / 2;
constexpr int64_t qsa_idx_s_at = qsa_idx_at + al(int64_t(640) * D * 2) / 2;
constexpr int64_t qsa_end = qsa_idx_s_at + al(int64_t(640) * 4) / 2;
constexpr int64_t ffn_down_at = al(conv_ffn) / 2;
constexpr int64_t ffn_router_at = ffn_down_at + al(int64_t(D) * 320 * 2) / 2;
constexpr int64_t ffn_router_s_at = ffn_router_at + al(int64_t(NE + 8) * D * 2) / 2;
constexpr int64_t ffn_end = ffn_router_s_at + al(int64_t(NE + 8) * 4) / 2;
//   the draft's catch-up (after layer 47 on the same streams): eh_proj [F][2 D], then wk and wv [256][D]; SP2's sparse
//   draft: then its index k (BF16 [128][D], its column scales after it)
constexpr int64_t draft_wk_at = al(int64_t(F) * 2 * D * 2) / 2;
constexpr int64_t draft_wv_at = draft_wk_at + al(int64_t(256) * D * 2) / 2;
constexpr int64_t draft_ik_at = draft_wv_at + al(int64_t(256) * D * 2) / 2;
constexpr int64_t draft_ik_s_at = draft_ik_at + al(int64_t(128) * D * 2) / 2;
constexpr int64_t draft_end = draft_ik_s_at + al(int64_t(128) * 4) / 2;

int64_t conv_bytes(bool single = false, int32_t card = 0) {
    return (single && card == 1 ? 0 : al(fn::hc::hct_weight_bytes())) +
           std::max({gdn_end, qsa_end, ffn_end, draft_end}) * 2;
}

// APF2: the ladder's split-K workspace a card (fn/kernels/tc/ladder.h lad_ws): partials and one counter a tile; the
// counters are zeroed at each chunk's start (the arena may be lent between chunks), each launch leaves them zero.
// The selector's cost model takes these limits, so a table entry always fits
constexpr int64_t lad_ws_floats = int64_t(6) << 20;  // 24 MiB
constexpr int32_t lad_ws_counts = 65536;
constexpr int64_t lad_ws_bytes  = al(lad_ws_floats * 4) + al(int64_t(lad_ws_counts) * 4);

// a bump allocator over a region
struct bump {
    uint8_t * p;
    int64_t   off = 0;

    template <typename T> T * take(int64_t bytes) {
        T * r = reinterpret_cast<T *>(p + off);
        off += al(bytes);
        return r;
    }
};

}  // namespace

// ---- the arena of one card ----
struct prompt_path::card_arena {
    float *             S         = nullptr;  // [RC][4][2560]
    float *             E         = nullptr;  // [RC][2560]
    int32_t *           tok       = nullptr;  // [RC]
    float * inject = nullptr;
    float * mixed = nullptr; // [RH][2560]
    uint16_t * mixed16 = nullptr;
    uint8_t *           p_send = nullptr, *p_recv = nullptr;  // MUP2: the P regions (p_msg)
    uint8_t *emb_send = nullptr, *emb_recv = nullptr;
    uint8_t *sel_send[2] = {}, *sel_recv[2] = {};
    uint8_t *           mom_send = nullptr, *mom_recv = nullptr;
    uint8_t *           dmix_send = nullptr, *dmix_recv = nullptr;
    float *             pred        = nullptr;  // [4][2560]
    float *             ple_hist[2] = {};       // [10240][9]
    float *             ple_stage   = nullptr;  // [5120][9] the peer's history
    uint32_t *          err         = nullptr;  // [0] prompt, [1] select, [2] EP1 status
    uint8_t *           scratch     = nullptr;
    uint8_t *           base        = nullptr;  // the arena's start (test_poison)
    // PB3's conversion region: the current sublayer's FP16 HC coefficients and dense weights
    fn::hc::hct_weights hct;
    uint16_t *          w16 = nullptr;
    fn::tc::lad_ws      lws;  // APF2: the ladder's split-K workspace

    // replicas: card 0's and card 1's halves (one of each pair is the resident region)
    struct hc_halves {
        const uint16_t * down[2];
        const uint16_t * up[2];
        const float *    gamma[2];
    };

    std::vector<hc_halves> hc_attn, hc_ffn;  // per layer
    const uint16_t *       ple_key[2] = {}, *ple_value[2] = {};
    const float *          ple_nk[2] = {}, *ple_nq[2] = {}, *ple_nc[2] = {};
    const uint16_t *       ple_conv[2] = {};
    float *                enorm       = nullptr;  // [2560] full
    float *                hnorm       = nullptr;  // [4][2560] full
    int32_t *              tok_host    = nullptr;  // pinned [2][RC]
    // APF1 Part 4: the prepared data (in the replicas' class, built once by prepare): per layer the HC sublayers' row
    // scales (hct_scales), and the BF16 products' column scales: GDN alpha | beta [48], the QSA index q | k [640]
    // (the index owner's), the router | shared gate | 7 zeros [520]; the draft's index k [128]
    std::vector<fn::hc::hct_weights> hc_prep_attn, hc_prep_ffn;
    std::vector<float *>             ab_s, idx_s, rt_s;
    float *                          dik_s = nullptr;
};

// one message link: card c sends its slot to card 1 - c's receive slot
struct prompt_path::link {
    uint8_t * send[2]   = { nullptr, nullptr };
    uint8_t * recv[2]   = { nullptr, nullptr };
    int64_t   bytes     = 0;
    void *    prod[2]   = { nullptr, nullptr };
    void *    copied[2] = { nullptr, nullptr };
    void *    freed[2]  = { nullptr, nullptr };
    uint64_t  g_prod[2] = { 0, 0 }, g_copied[2] = { 0, 0 }, g_freed[2] = { 0, 0 };
    bool      any_copied[2] = { false, false }, any_freed[2] = { false, false };
};

namespace {

// MUP2: a P link a piece (its events and generations; its slot's place is the chunk's, p_msg)
enum link_id : int32_t { lk_emb, lk_sel0, lk_sel1, lk_mom, lk_dmix, lk_p0, n_links = lk_p0 + prompt_max_pieces };

int64_t arena_bytes_a(int64_t rc, int64_t rh, bool single, int32_t card) {
    if (!single) {
        int64_t n = 0;
        n += al(rc * 4 * D * 4) + al(rc * D * 4) + al(rc * 4) + al(rc * 4 * 4);
        n += al(rh * D * 4) + al(rh * D * 2) + al(rh * D * 4);
        n += 2 * al(p_region(rh)) + 2 * al(emb_slot(rc)) + 4 * al(sel_slot) + 2 * al(mom_slot(rh)) +
             2 * al(dmix_slot(rh));
        n += al(int64_t(4) * D * 4) + 2 * al(int64_t(4) * D * 9 * 4) + al(ple_hist_card * 4) + al(64);
        n += al(family_bytes(rh)) + al(conv_bytes()) + lad_ws_bytes;
        return n;
    }
    int64_t n = 0;
    if (card == 0) {
        n += al(rc * 4 * D * 4) + al(rc * D * 4) + al(rc * 4) + al(rc * 4 * 4);
        n += al(rh * D * 2);
    }
    n += al(p_region(rh)) + 2 * al(sel_slot) + al(mom_slot(rh)) + al(dmix_slot(rh));
    if (card == 0) {
        n += al(int64_t(4) * D * 4) + 2 * al(int64_t(4) * D * 9 * 4);
    }
    n += al(64) + al(family_bytes_single(rc, rh, card)) + al(conv_bytes(true, card)) + lad_ws_bytes;
    return n;
}

} // namespace

int64_t prompt_arena_bytes(int32_t chunk_rows, int32_t half_rows, bool single, int32_t card) {
    return arena_bytes_a(chunk_rows, half_rows, single, card);
}

namespace {

// APF1 Part 4: the prepared data's bytes a card (prompt_path::prepare; card_arena's hc_prep_*, ab_s .. dik_s)
constexpr int64_t prep_layer_bytes = al(int64_t(48) * 4) + al(int64_t(640) * 4) + al(int64_t(NE + 8) * 4);

int64_t prep_bytes() {
    return int64_t(2) * fn::n_layer * fn::hc::hct_scale_bytes() + int64_t(fn::n_layer) * prep_layer_bytes +
           al(int64_t(128) * 4);
}

}  // namespace

int64_t prompt_replica_bytes(bool single, fn::ple::kv_fmt ple_kv, bool ple_conv32) {
    const int64_t norms = al(int64_t(D) * 4) + al(int64_t(4) * D * 4);
    if (single) {
        return norms + prep_bytes();
    }
    return int64_t(2) * fn::n_layer * al(hc_down_bytes + hc_up_bytes + hc_gamma_bytes) + al(ple_key_bytes_of(ple_kv)) +
           al(ple_value_bytes_of(ple_kv)) + 3 * al(ple_norm_bytes) + al(ple_conv_bytes_of(ple_conv32)) + norms +
           prep_bytes();
}

namespace {

std::mutex                totals_mu;
prompt_path::route_counts totals;
uint32_t                  totals_rungs = 0;
std::atomic<bool>         totals_on{ false };  // test-only collection (prompt_route_collect); off in serving

constexpr int64_t prompt_path::route_counts::* count_fields[] = {
    &prompt_path::route_counts::dense_ladder,  &prompt_path::route_counts::dense_retained,
    &prompt_path::route_counts::dense_split,   &prompt_path::route_counts::hc_ladder,
    &prompt_path::route_counts::hc_retained,   &prompt_path::route_counts::ex_grouped,
    &prompt_path::route_counts::ex_per_task,   &prompt_path::route_counts::exd_grouped,
    &prompt_path::route_counts::exd_per_task,  &prompt_path::route_counts::graph_units,
    &prompt_path::route_counts::eager_units,   &prompt_path::route_counts::graph_captures,
    &prompt_path::route_counts::graph_updated, &prompt_path::route_counts::qsa,
    &prompt_path::route_counts::gdn,           &prompt_path::route_counts::ple,
};

// the totals less what a failed capture noted (now - before)
void undo_totals(const prompt_path::route_counts & now, const prompt_path::route_counts & before) {
    if (!totals_on.load(std::memory_order_relaxed)) {
        return;
    }
    const std::lock_guard<std::mutex> g(totals_mu);
    for (const auto f : count_fields) {
        totals.*f -= now.*f - before.*f;
    }
}

}  // namespace

void prompt_route_totals(prompt_path::route_counts & out, uint32_t & rungs, bool clear_fmts) {
    const std::lock_guard<std::mutex> g(totals_mu);
    out   = totals;
    rungs = totals_rungs;
    if (clear_fmts) {
        totals.dense_fmts = 0;
        totals.ex_fmts    = 0;
    }
}

void prompt_route_collect(bool on) {
    totals_on.store(on, std::memory_order_relaxed);
}

void prompt_path::note(int64_t route_counts::* f) {
    ++(rcount_.*f);
    if (!totals_on.load(std::memory_order_relaxed)) {
        return;
    }
    const std::lock_guard<std::mutex> g(totals_mu);
    ++(totals.*f);
}

void prompt_path::note_fmt(uint32_t route_counts::* f, int32_t bit) {
    rcount_.*f |= 1u << bit;
    if (!totals_on.load(std::memory_order_relaxed)) {
        return;
    }
    const std::lock_guard<std::mutex> g(totals_mu);
    totals.*f |= 1u << bit;
}

void prompt_path::route_counts::add_since(const route_counts & a, const route_counts & b) {
    for (const auto f : count_fields) {
        this->*f += a.*f - b.*f;
    }
    dense_fmts |= a.dense_fmts;
    ex_fmts |= a.ex_fmts;
}

std::string prompt_path::route_counts::text() const {
    const auto names = [](uint32_t bits, bool dense) {
        std::string t;
        for (int32_t i = 0; i < 32; ++i) {
            if ((bits >> i & 1u) != 0) {
                t += (t.empty() ? " (" : " ") +
                     std::string(dense ? fn::dense::fmt_name(static_cast<fn::dense::fmt>(i)) :
                                         fn::experts::fmt_name(static_cast<fn::experts::fmt>(i)));
            }
        }
        return t.empty() ? t : t + ")";
    };
    const auto n = [](int64_t v) {
        return std::to_string(v);
    };
    return "dense ladder " + n(dense_ladder) + names(dense_fmts, true) + " (split " + n(dense_split) + ") / retained " +
           n(dense_retained) + ", hc ladder " + n(hc_ladder) + " / retained " + n(hc_retained) +
           ", experts gate/up grouped " + n(ex_grouped) + " / per-task " + n(ex_per_task) + names(ex_fmts, false) +
           ", down grouped " + n(exd_grouped) + " / per-task " + n(exd_per_task) + ", qsa " + n(qsa) + ", gdn " +
           n(gdn) + ", ple " + n(ple) + " (retained), units graph " + n(graph_units) + " (captures " +
           n(graph_captures) + ", updated " + n(graph_updated) + ") / eager " + n(eager_units);
}

bool prompt_path::test_skew_experts(bool on, std::string & err) {
    if (on && skew_host_ == nullptr) {
        void * h = nullptr;
        if (!cuda_ok(cudaHostAlloc(&h, size_t(RC) * NU * 4, cudaHostAllocPortable), "skew rows", err)) {
            return false;
        }
        skew_host_ = static_cast<int32_t *>(h);
        for (int32_t r = 0; r < RC; ++r) {
            for (int32_t q = 0; q < NU; ++q) {
                skew_host_[int64_t(r) * NU + q] = q < 4 ? q : 4 + (37 * r + 61 * q) % 400;
            }
        }
    }
    skew_ = on;
    return true;
}

namespace {

std::atomic<bool> graphs_on{ true };  // APF2: the graph switch (the server's props)

}  // namespace

void prompt_graphs(bool on) {
    graphs_on.store(on, std::memory_order_relaxed);
}

bool prompt_graphs_on() {
    return graphs_on.load(std::memory_order_relaxed);
}

prompt_route prompt_route_for(int32_t rows, bool graph_ok, const prompt_route_force & force) {
    prompt_route r;
    r.rung = 64;
    while (r.rung < rows && r.rung < prompt_chunk_rows) {
        r.rung *= 2;
    }
    const bool dflt = prompt_graphs_on() && rows <= prompt_graph_rows;
    r.graph         = (force.graph == 1 ? true : force.graph == 2 ? false : dflt) && graph_ok;
    return r;
}

std::string prompt_route_name(const prompt_route & r) {
    return "rung " + std::to_string(r.rung) + ", " + (r.graph ? "graph" : "eager");
}

// ---- APF2: the ladder's selection -----------------------------------------------------------------------------------
namespace {

// the weights' bytes a value (TurboMind's type_b) by the key's format name
double lad_b_bytes(const std::string & f) {
    static const std::pair<const char *, double> t[] = {
        { "BF16",       2.0         },
        { "Q8_0",       34.0 / 32   },
        { "Q8_0/EX1",   34.0 / 32   },
        { "Q4_K",       144.0 / 256 },
        { "Q5_K",       176.0 / 256 },
        { "Q6_K",       210.0 / 256 },
        { "IQ4_XS",     136.0 / 256 },
        { "IQ3_S",      110.0 / 256 },
        { "IQ4_NL/EX1", 18.0 / 32   },
        { "Q2_0/EX1",   10.0 / 32   },
        { "Q5_1",       24.0 / 32   }
    };
    for (const auto & [n, b] : t) {
        if (f == n) {
            return b;
        }
    }
    return 1.0;
}

}  // namespace

fn::model::lad_key prompt_dense_key(int32_t f, int32_t lay, int32_t k, int32_t n) {
    const auto df = static_cast<fn::dense::fmt>(f);
    return { fn::model::lad_fam::dense,
             std::string(fn::dense::fmt_name(df)) +
                 (static_cast<fn::dense::pd_layout>(lay) == fn::dense::pd_layout::ex1_down ? "/EX1" : ""),
             k, n };
}

fn::model::lad_key prompt_hc_key(bool up, bool q8) {
    return { up ? fn::model::lad_fam::hc_up : fn::model::lad_fam::hc_down, q8 ? "Q8_0" : "BF16",
             up ? fn::hc::hcl_up_k : fn::hc::hcl_down_k, up ? fn::hc::hcl_up_n : fn::hc::hcl_down_n };
}

fn::model::lad_key prompt_expert_key(bool down, int32_t f) {
    return { down ? fn::model::lad_fam::ex_down : fn::model::lad_fam::ex_gate_up,
             fn::experts::fmt_name(static_cast<fn::experts::fmt>(f)), down ? 320 : D, down ? D : 640 };
}

int32_t prompt_expert_rows(int32_t rows) {
    return rows * NU / NE;
}

bool prompt_lad_model(const fn::model::lad_key & key, int32_t device, fn::model::lad_model & m) {
    namespace fm = fn::model;
    m            = {};
    if (cudaDeviceGetAttribute(&m.sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        return false;
    }
    m.b_bytes = lad_b_bytes(key.fmt);
    if (key.fam == fm::lad_fam::dense || key.fam == fm::lad_fam::hc_down || key.fam == fm::lad_fam::hc_up) {
        m.ws_floats = lad_ws_floats;
        m.ws_count  = lad_ws_counts;
    }
    m.num = key.fam == fm::lad_fam::hc_down                                       ? 4 :
            key.fam == fm::lad_fam::ex_gate_up || key.fam == fm::lad_fam::ex_down ? NE :
                                                                                    1;
    for (int32_t t = 0; t < fm::lad_fam_tiles(key.fam); ++t) {
        fn::tc::lad_resources r;
        bool                  ok = false;
        if (key.fam == fm::lad_fam::dense) {
            fn::dense::pq_args q;
            const bool         ex1 = key.fmt.size() > 4 && key.fmt.compare(key.fmt.size() - 4, 4, "/EX1") == 0;
            const std::string  fn_ = ex1 ? key.fmt.substr(0, key.fmt.size() - 4) : key.fmt;
            q.lay                  = ex1 ? fn::dense::pd_layout::ex1_down : fn::dense::pd_layout::dp2;
            q.n_src                = fn_ == "BF16" ? 0 : 1;
            for (int32_t f = 0; f < 32 && q.n_src > 0; ++f) {
                if (fn_ == fn::dense::fmt_name(static_cast<fn::dense::fmt>(f))) {
                    q.f = static_cast<fn::dense::fmt>(f);
                }
            }
            fn::dense::pd_resources pr{};
            ok            = fn::dense::pl_resources(q, t, pr);
            r.ctas_per_sm = pr.ctas_per_sm;
        } else if (key.fam == fm::lad_fam::hc_down || key.fam == fm::lad_fam::hc_up) {
            ok = fn::hc::hcl_resources(key.fmt == "Q8_0", key.fam == fm::lad_fam::hc_up ? 1 : 0, t, r);
        } else {
            for (int32_t f = 0; f < 32; ++f) {
                if (key.fmt == fn::experts::fmt_name(static_cast<fn::experts::fmt>(f))) {
                    ok = fn::experts::el_resources(key.fam == fm::lad_fam::ex_down ? 1 : 0,
                                                   static_cast<fn::experts::fmt>(f), t, r);
                }
            }
        }
        m.ctas[t] = ok ? r.ctas_per_sm : 0;
    }
    (void) cudaGetLastError();  // a format a family does not compile: no CTA (the filter's reason)
    return true;
}

const fn::model::lad_model & prompt_path::lad_model_of(const fn::model::lad_key & key) {
    for (const auto & [k, m] : lad_models_) {
        if (k == key) {
            return m;
        }
    }
    fn::model::lad_model m;
    (void) prompt_lad_model(key, m_ != nullptr ? m_->device[0] : 0, m);
    lad_models_.emplace_back(key, m);
    return lad_models_.back().second;
}

fn::model::lad_cfg prompt_path::lad_route(const fn::model::lad_key & key, int32_t rows, int32_t force) {
    if (force == 2) {
        return {};
    }
    if (force == 1) {
        const fn::model::lad_ranking r =
            fn::model::lad_rank(key.fam, std::max(1, rows), key.k, key.n, lad_model_of(key));
        return r.ranked.empty() ? fn::model::lad_cfg{} : r.ranked.front().cfg;
    }
    // the table is the UD-Q4_K_XL file's (prompt-tune's calibration): another file (IQ3_XXS) runs the retained routes
    // wherever its keys would match. The pick at the rung, the retained route where its split does not fit the
    // workspace at these rows
    return lad_on_ ? fn::model::lad_lookup(key, rows, lad_model_of(key)) : fn::model::lad_cfg{};
}

bool prompt_path::lad_retained_any(const fn::model::lad_key & key,
                                   const prompt_chunk &       c,
                                   int32_t                    force,
                                   int32_t                    mul,
                                   bool                       experts) {
    for (int32_t j = 0; j < prompt_pieces(c); ++j) {
        const int32_t rows = prompt_piece_rows(c, j) * mul;
        if (lad_route(key, experts ? prompt_expert_rows(rows) : rows, force).retained()) {
            return true;
        }
    }
    return false;
}

namespace {

// TTTS2: an HC module's halves as the prompt path reads them: BF16, or hc.h's Q8_0 layouts (UD-Q4_K_XL)
const uint16_t * hc_down_of(const hc_module & h) {
    return h.q8_down != nullptr ? reinterpret_cast<const uint16_t *>(h.q8_down) : h.w_down;
}

const uint16_t * hc_up_of(const hc_module & h) {
    return h.q8_up != nullptr ? reinterpret_cast<const uint16_t *>(h.q8_up) : h.w_up;
}

int64_t hc_down_bytes_of(const hc_module & h) {
    return h.q8_down == nullptr ? hc_down_bytes : h.q8_mix ? fn::hc::w_down_mix_bytes : fn::hc::w_down_q8_bytes;
}

int64_t hc_up_bytes_of(const hc_module & h) {
    return h.q8_up == nullptr ? hc_up_bytes : fn::hc::w_up_q8_bytes;
}

}  // namespace

int32_t prompt_pieces(const prompt_chunk & c) {
    return c.n_pieces > 0 ? c.n_pieces : c.half[1] > 0 ? 2 : 1;
}

int32_t prompt_piece_r0(const prompt_chunk & c, int32_t j) {
    if (c.n_pieces <= 0) {
        return j == 0 ? 0 : c.half[0];
    }
    int32_t r0 = 0;
    for (int32_t i = 0; i < j; ++i) {
        r0 += c.piece[i];
    }
    return r0;
}

int32_t prompt_piece_rows(const prompt_chunk & c, int32_t j) {
    return c.n_pieces > 0 ? c.piece[j] : c.half[j];
}

bool prompt_set_pieces(prompt_chunk & c, int32_t rows, std::string & err, int32_t min_rows) {
    c.n_pieces = 0;
    if (rows == 0) {
        return true;
    }
    if (rows < prompt_piece_grid || rows % prompt_piece_grid != 0 || min_rows < 0) {
        return fail(err, "prompt: pieces of " + std::to_string(rows) + " rows (a positive multiple of 64), at least " +
                             std::to_string(min_rows));
    }
    // each half in the fewest pieces of at most rows rows, evened out on the grid (no short tail piece paying the
    // fixed cost for a few rows): k pieces of ceil(half / k) rounded up to 64, the last the rest. PRD1 item 12: with
    // min_rows, no piece made by subdividing a half has fewer rows. At each k, from the fewest pieces of at most rows
    // down to 2: the evened pieces when their last has min_rows; else k - 1 pieces of floor(half / k) rounded down to
    // 64, when that has min_rows, and the last the larger rest; else one piece fewer. At k = 1 the half stays whole
    int32_t n = 0;
    for (int32_t h = 0; h < 2 && c.half[h] > 0; ++h) {
        const int32_t g    = prompt_piece_grid;
        const int32_t half = c.half[h];
        const auto    even = [&](int32_t k) {
            return ((half + k - 1) / k + g - 1) / g * g;
        };
        int32_t k    = (half + rows - 1) / rows;
        int32_t e    = even(k);
        bool    rest = false;  // k - 1 pieces of e and the rest (above e) last
        for (; min_rows > 0 && k > 1; --k) {
            e = even(k);
            if (half - (k - 1) * e >= min_rows) {
                break;
            }
            e = half / k / g * g;
            if (e >= min_rows) {
                rest = true;
                break;
            }
        }
        if (min_rows > 0 && k == 1) {
            e = half;
        }
        for (int32_t left = half, i = 0; left > 0; ++i) {
            if (n == prompt_max_pieces) {
                c.n_pieces = 0;
                return fail(err, "prompt: a chunk of " + std::to_string(c.rows) + " rows in pieces of " +
                                     std::to_string(rows) + " rows: more than " + std::to_string(prompt_max_pieces) +
                                     " pieces");
            }
            const int32_t p = rest && i == k - 1 ? left : std::min(left, e);
            c.piece[n++]    = p;
            left -= p;
        }
    }
    c.n_pieces = n;
    return true;
}

bool prompt_pieces_ok(const prompt_chunk & c) {
    if (c.n_pieces == 0) {
        return true;
    }
    if (c.n_pieces < 1 || c.n_pieces > prompt_max_pieces) {
        return false;
    }
    int32_t r0          = 0;
    bool    cut_at_half = c.half[1] == 0;
    for (int32_t j = 0; j < c.n_pieces; ++j) {
        if (c.piece[j] < 1 || (j + 1 < c.n_pieces && (r0 + c.piece[j]) % prompt_piece_grid != 0)) {
            return false;
        }
        r0 += c.piece[j];
        cut_at_half = cut_at_half || r0 == c.half[0];
    }
    return r0 == c.rows && cut_at_half;
}

const char * psite_name(psite s) {
    static const char * const names[n_psites] = { "S01 embed",      "S02 hc.attn", "S03 hc.ffn",     "S04 gdn.input",
                                                  "S05 gdn.core",   "S06 gdn.out", "S07 qsa.input",  "S08 qsa.index",
                                                  "S09 qsa.attend", "S10 qsa.out", "S11 ffn.router", "S12 ffn.experts",
                                                  "S13 ffn.shared", "S14 ple",     "S15 exchange",   "S16 draft",
                                                  "S17 head" };
    const int32_t             i               = static_cast<int32_t>(s);
    return i >= 0 && i < n_psites ? names[i] : "?";
}

prompt_path::prompt_path() = default;

// APF2: a unit's graph: its executable and its operations in capture order (empty ops: not replayable by update)
// (the captured graph kept with it: the noted nodes are the graph's, which the executable's updates name)
struct prompt_path::unit_graph_entry {
    cudaGraph_t                     graph = nullptr;
    cudaGraphExec_t                 exec  = nullptr;
    std::vector<fn::model::grec_op> ops;
    bool                            updatable = false;

    ~unit_graph_entry() {
        if (exec != nullptr) {
            cudaGraphExecDestroy(exec);
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
    }
};

prompt_path::~prompt_path() {
    ugraphs_.clear();
    for (int32_t c = 0; c < 2 && m_ != nullptr; ++c) {
        cudaSetDevice(m_->device[c]);
        for (void * e : gj_[c]) {
            if (e != nullptr) {
                cudaEventDestroy(static_cast<cudaEvent_t>(e));
            }
        }
        if (copy_[c] != nullptr) {
            cudaStreamSynchronize(static_cast<cudaStream_t>(copy_[c]));
            cudaStreamDestroy(static_cast<cudaStream_t>(copy_[c]));
        }
        for (link & l : links_) {
            for (void * e : { l.prod[c], l.copied[c], l.freed[c] }) {
                if (e != nullptr) {
                    cudaEventDestroy(static_cast<cudaEvent_t>(e));
                }
            }
        }
    }
    for (card_arena & a : a_) {
        if (a.tok_host != nullptr) {
            cudaFreeHost(a.tok_host);
        }
    }
    if (skew_host_ != nullptr) {
        cudaFreeHost(skew_host_);
    }
}

namespace {

uint8_t * class_part0(const model & m, const char * cls, int32_t card) {
    const reserved_class * rc = m.state_class(cls);
    return rc == nullptr || rc->plan.parts.empty() ? nullptr : rc->part_ptr(0, card);
}

}  // namespace

// the arena's row-dependent views for chunks of rc rows in halves of rh, from its base: every view the prompt writes
// lies in [base, base + prompt_arena_bytes(rc, rh)) (ME3: a smaller layout is a prefix, so a loan sized to it covers
// every write)
bool prompt_path::carve(card_arena & a, int32_t rc, int32_t rh, bool single, int32_t card, std::string & err) {
    bump b{a.base};
    if (!single || card == 0) {
        a.S = b.take<float>(int64_t(rc) * 4 * D * 4);
        a.E = b.take<float>(int64_t(rc) * D * 4);
        a.tok = b.take<int32_t>(int64_t(rc) * 4);
        a.inject = b.take<float>(int64_t(rc) * 4 * 4);
        if (!single) {
            a.mixed = b.take<float>(int64_t(rh) * D * 4);
        }
        a.mixed16 = b.take<uint16_t>(int64_t(rh) * D * 2);
        if (!single) {
            (void)b.take<float>(int64_t(rh) * D * 4);
        }
    }
    a.p_send = b.take<uint8_t>(p_region(rh));
    if (!single) {
        a.p_recv = b.take<uint8_t>(p_region(rh));
    }
    if (!single) {
        a.emb_send = b.take<uint8_t>(emb_slot(rc));
        a.emb_recv = b.take<uint8_t>(emb_slot(rc));
    }
    for (int32_t h = 0; h < 2; ++h) {
        a.sel_send[h] = b.take<uint8_t>(sel_slot);
        if (!single) {
            a.sel_recv[h] = b.take<uint8_t>(sel_slot);
        }
    }
    a.mom_send = b.take<uint8_t>(mom_slot(rh));
    if (!single) {
        a.mom_recv = b.take<uint8_t>(mom_slot(rh));
    }
    a.dmix_send = b.take<uint8_t>(dmix_slot(rh));
    if (!single) {
        a.dmix_recv = b.take<uint8_t>(dmix_slot(rh));
    }
    if (!single || card == 0) {
        a.pred = b.take<float>(int64_t(4) * D * 4);
        a.ple_hist[0] = b.take<float>(int64_t(4) * D * 9 * 4);
        a.ple_hist[1] = b.take<float>(int64_t(4) * D * 9 * 4);
        if (!single) {
            a.ple_stage = b.take<float>(ple_hist_card * 4);
        }
    }
    a.err = b.take<uint32_t>(64);
    a.scratch = b.take<uint8_t>(single ? family_bytes_single(rc, rh, card) : family_bytes(rh));
    if (single) {
        a.emb_send = a.scratch;
        if (card == 0) {
            a.mixed = reinterpret_cast<float *>(a.scratch + hc_base_bytes(rh));
        }
    }
    uint8_t * cv = b.take<uint8_t>(conv_bytes(single, card));
    if (!single || card == 0) {
        a.hct = fn::hc::hct_carve(cv);
        a.w16 = reinterpret_cast<uint16_t *>(cv + al(fn::hc::hct_weight_bytes()));
    } else {
        a.hct = {};
        a.w16 = reinterpret_cast<uint16_t *>(cv);
    }
    a.lws.part        = b.take<float>(lad_ws_floats * 4);
    a.lws.part_floats = lad_ws_floats;
    a.lws.count       = b.take<int32_t>(int64_t(lad_ws_counts) * 4);
    a.lws.count_len   = lad_ws_counts;
    if (b.off != prompt_arena_bytes(rc, rh, single, card) ||
        b.off > prompt_arena_bytes(rc_max_, rh_max_, single, card)) {
        return fail(err, "prompt: the arena's carve disagrees with its size");
    }
    return true;
}

void prompt_path::set_links() {
    if (m_ != nullptr && m_->single) {
        a_[1].S = a_[0].S;
        a_[1].E = a_[0].E;
        a_[1].tok = a_[0].tok;
        a_[1].inject = a_[0].inject;
        a_[1].mixed = a_[0].mixed;
        a_[1].mixed16 = a_[0].mixed16;
        a_[1].pred = a_[0].pred;
        a_[1].ple_hist[0] = a_[0].ple_hist[0];
        a_[1].ple_hist[1] = a_[0].ple_hist[1];
        a_[0].ple_stage = nullptr;
        a_[1].ple_stage = nullptr;
        for (int32_t c = 0; c < 2; ++c) {
            card_arena & a = a_[size_t(c)];
            const card_arena & p = a_[size_t(1 - c)];
            a.p_recv             = p.p_send;
            for (int32_t h = 0; h < 2; ++h) {
                a.sel_recv[h] = p.sel_send[h];
            }
            a.emb_recv = p.emb_send;
            a.mom_recv = p.mom_send;
            a.dmix_recv = p.dmix_send;
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        card_arena & a          = a_[size_t(c)];
        links_[lk_emb].send[c]  = a.emb_send;
        links_[lk_emb].recv[c]  = a.emb_recv;
        links_[lk_sel0].send[c] = a.sel_send[0];
        links_[lk_sel0].recv[c] = a.sel_recv[0];
        links_[lk_sel1].send[c] = a.sel_send[1];
        links_[lk_sel1].recv[c] = a.sel_recv[1];
        links_[lk_mom].send[c]  = a.mom_send;
        links_[lk_mom].recv[c]  = a.mom_recv;
        links_[lk_dmix].send[c] = a.dmix_send;
        links_[lk_dmix].recv[c] = a.dmix_recv;
    }
}

bool prompt_path::layout(int32_t rc, int32_t rh, std::string & err) {
    if (rc < 1 || rc > rc_max_ || rh < 1 || rh > rh_max_ || (held_ && !held_whole_)) {
        return fail(err, "prompt: a layout outside 1 .. 8,192 rows in halves of at most 4,096, or under a sized loan");
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!carve(a_[size_t(c)], rc, rh, m_->single, c, err)) {
            return false;
        }
    }
    set_links();
    rc_ = rc;
    rh_ = rh;
    return true;
}

void prompt_path::bind(pass_card * const p[2], draft_card * const d[2]) {
    for (int32_t c = 0; c < 2; ++c) {
        p_[c] = p[c];
        d_[c] = d[c];
    }
}

void prompt_path::arena_spans(std::vector<byte_span> & out) const {
    for (int32_t c = 0; c < 2; ++c) {
        out.push_back({a_[size_t(c)].base, prompt_arena_bytes(rc_, rh_, m_->single, c)});
    }
}

expert_service * prompt_path::service() const {
    return p_[0] != nullptr ? p_[0]->c.ex_service : nullptr;
}

bool prompt_path::loan_check(const std::vector<byte_span> & need, std::string & err) const {
    expert_service * x = p_[0]->c.ex_service;
    return x == nullptr || x->prompt_loan_check(need, err);
}

bool prompt_path::init(model & m, pass_card * const p[2], draft_card * const d[2], std::string & err,
                       const char * arena_class, int32_t rc_max, int32_t rh_max) {
    m_ = &m;
    // APF2: whether the measured table applies, once: dense_q8_of scans the tensor table 336 times, far too slow for
    // every route lookup (about a second a read)
    lad_on_ = dense_q8_of(m);
    a_.assign(2, card_arena());
    if (rc_max < 1 || rc_max > RC || rh_max < 1 || rh_max > RH) {
        return fail(err, "prompt: an arena layout outside 8,192 rows in halves of 4,096");
    }
    rc_max_ = rc_max;
    rh_max_ = rh_max;
    for (int32_t c = 0; c < 2; ++c) {
        p_[c]          = p[c];
        d_[c]          = d[c];
        uint8_t * base = class_part0(m, arena_class, c);
        uint8_t * rep  = class_part0(m, "HC and PLE replicas (PB1)", c);
        const reserved_class * ac = m.state_class(arena_class);
        if (base == nullptr || rep == nullptr ||
            ac->plan.parts[0].bytes[c] < prompt_arena_bytes(rc_max, rh_max, m.single, c)) {
            return fail(err, std::string("prompt: the arena (") + arena_class +
                                 ") or the replicas are not reserved at their size (PB1)");
        }
        card_arena & a = a_[size_t(c)];
        a.base = base;
        if (!carve(a, rc_max, rh_max, m.single, c, err)) {
            return false;
        }
        // the replicas' places (filled below): per layer and sublayer the peer's down, up and gamma, then PLE's, then the
        // draft's full norms; ME1, one device: the peer's resident HC and PLE tensors themselves, no copy
        bump r{ rep };
        const bool one = m.single;
        a.hc_attn.resize(fn::n_layer);
        a.hc_ffn.resize(fn::n_layer);
        for (int32_t l = 0; l < fn::n_layer; ++l) {
            for (int32_t sub = 0; sub < 2; ++sub) {
                card_arena::hc_halves & hh  = sub == 0 ? a.hc_attn[size_t(l)] : a.hc_ffn[size_t(l)];
                const hc_module &       own = sub == 0 ? p[c]->w[l].hc_attn : p[c]->w[l].hc_ffn;
                hh.down[c]                  = hc_down_of(own);
                hh.up[c]                    = hc_up_of(own);
                hh.gamma[c]                 = own.gamma;
                if (one) {
                    const hc_module & pw = sub == 0 ? p[1 - c]->w[l].hc_attn : p[1 - c]->w[l].hc_ffn;
                    hh.down[1 - c]       = hc_down_of(pw);
                    hh.up[1 - c]         = hc_up_of(pw);
                    hh.gamma[1 - c]      = pw.gamma;
                    continue;
                }
                uint8_t * blk   = r.take<uint8_t>(hc_down_bytes + hc_up_bytes + hc_gamma_bytes);
                hh.down[1 - c]  = reinterpret_cast<const uint16_t *>(blk);
                hh.up[1 - c]    = reinterpret_cast<const uint16_t *>(blk + hc_down_bytes);
                hh.gamma[1 - c] = reinterpret_cast<const float *>(blk + hc_down_bytes + hc_up_bytes);
            }
        }
        a.ple_key[c]       = reinterpret_cast<const uint16_t *>(p[c]->ple_w.key);
        a.ple_value[c]     = reinterpret_cast<const uint16_t *>(p[c]->ple_w.value);
        a.ple_nk[c]        = p[c]->ple_w.norm_key;
        a.ple_nq[c]        = p[c]->ple_w.norm_query;
        a.ple_nc[c]        = p[c]->ple_w.norm_conv;
        a.ple_conv[c]      = p[c]->ple_w.conv;  // TTTS2: F32 when ple_w.conv32 (the same pointer)
        if (one) {
            const fn::ple::weights & pw = p[1 - c]->ple_w;
            a.ple_key[1 - c]            = reinterpret_cast<const uint16_t *>(pw.key);
            a.ple_value[1 - c]          = reinterpret_cast<const uint16_t *>(pw.value);
            a.ple_nk[1 - c]             = pw.norm_key;
            a.ple_nq[1 - c]             = pw.norm_query;
            a.ple_nc[1 - c]             = pw.norm_conv;
            a.ple_conv[1 - c]           = pw.conv;
        } else {
            // TTTS3: the key and value in the file's format, TTTS2's F32 taps; the file's rule (as the reservation's,
            // run_load file_state_params), since a box without PLE in its window leaves its ple_w unset
            a.ple_key[1 - c]   = r.take<const uint16_t>(ple_key_bytes_of(ple_kv_of(m)));
            a.ple_value[1 - c] = r.take<const uint16_t>(ple_value_bytes_of(ple_kv_of(m)));
            a.ple_nk[1 - c]    = r.take<const float>(ple_norm_bytes);
            a.ple_nq[1 - c]    = r.take<const float>(ple_norm_bytes);
            a.ple_nc[1 - c]    = r.take<const float>(ple_norm_bytes);
            a.ple_conv[1 - c]  = r.take<const uint16_t>(ple_conv_bytes_of(ple_conv32_of(m)));
        }
        a.enorm = r.take<float>(int64_t(D) * 4);
        a.hnorm = r.take<float>(int64_t(4) * D * 4);
        // APF1 Part 4: the prepared data's places (prep_bytes; filled by prepare)
        a.hc_prep_attn.resize(fn::n_layer);
        a.hc_prep_ffn.resize(fn::n_layer);
        for (int32_t l = 0; l < fn::n_layer; ++l) {
            a.hc_prep_attn[size_t(l)] = fn::hc::hct_carve_scales(r.take<uint8_t>(fn::hc::hct_scale_bytes()));
            a.hc_prep_ffn[size_t(l)]  = fn::hc::hct_carve_scales(r.take<uint8_t>(fn::hc::hct_scale_bytes()));
        }
        for (int32_t l = 0; l < fn::n_layer; ++l) {
            a.ab_s.push_back(r.take<float>(int64_t(48) * 4));
            a.idx_s.push_back(r.take<float>(int64_t(640) * 4));
            a.rt_s.push_back(r.take<float>(int64_t(NE + 8) * 4));
        }
        a.dik_s                    = r.take<float>(int64_t(128) * 4);
        // TTTS3: against the part as reserved, not a size recomputed from the carve's own inputs
        const reserved_class * rpc = m.state_class("HC and PLE replicas (PB1)");
        if (r.off > rpc->plan.parts[0].bytes[c]) {
            return fail(err, "prompt: the replicas' carve exceeds their reservation");
        }
        void * th = nullptr;
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device", err) ||
            !cuda_ok(cudaHostAlloc(&th, size_t(2) * RC * 4, cudaHostAllocPortable), "pinned tokens", err)) {
            return false;
        }
        a.tok_host      = static_cast<int32_t *>(th);
        // ME2b: a lent arena holds expert slots now (the pool's); its error words are cleared at each borrow (hold)
        const bool lent = p[0]->c.ex_service != nullptr && p[0]->c.ex_service->prompt_lent();
        cudaStream_t cs = nullptr;
        if (!cuda_ok(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "copy stream", err) ||
            (!lent && !cuda_ok(cudaMemset(a.err, 0, 64), "error words", err))) {
            return false;
        }
        copy_[c] = cs;
    }
    rc_ = rc_max;
    rh_ = rh_max;
    links_.assign(n_links, link());
    set_links();
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device", err)) {
            return false;
        }
        for (link & l : links_) {
            for (void ** e : { &l.prod[c], &l.copied[c], &l.freed[c] }) {
                cudaEvent_t ev = nullptr;
                if (!cuda_ok(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "event", err)) {
                    return false;
                }
                *e = ev;
            }
        }
    }
    return fill_replicas(err) && prepare(err);
}

// APF1 Part 4: the ladder's prepared data, once at load: each HC sublayer's row scales (hct_scales, from both halves:
// after fill_replicas) and the BF16 products' column scales (pd_bf16_scales), each bitwise the converted route's
// scales; the window's layers only (TTTS1). Synchronous
bool prompt_path::prepare(std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        card_arena &      a = a_[size_t(c)];
        const pass_card & p = *p_[c];
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err)) {
            return false;
        }
        cudaStream_t s = static_cast<cudaStream_t>(p.c.stream);
        // the kernels' shared memory opt-ins on this device now, so a unit's capture sets no attribute
        if (!fn::dense::pq_tc_attributes() || !fn::experts::ep_prompt_attributes() ||
            !fn::gdn::gdn_prompt_attributes() || !fn::dense::pl_attributes() || !fn::hc::hcl_attributes()) {
            return fail(err, "prompt: a kernel's shared memory opt-in");
        }
        for (int32_t l = 0; l < fn::n_layer; ++l) {
            if (!m_->win.layer(l)) {
                continue;
            }
            const layer_weights & w = p.w[l];
            for (int32_t sub = 0; sub < 2; ++sub) {
                const card_arena::hc_halves & hh = sub == 0 ? a.hc_attn[size_t(l)] : a.hc_ffn[size_t(l)];
                fn::hc::hcp_weights           hw;
                hw.form = fn::hc::hcp_form::replicated;
                hw.q8   = (sub == 0 ? w.hc_attn : w.hc_ffn).q8_down != nullptr;
                for (int32_t q = 0; q < 2; ++q) {
                    hw.down[q]  = hh.down[q];
                    hw.up[q]    = hh.up[q];
                    hw.gamma[q] = hh.gamma[q];
                }
                if (!fn::hc::hct_scales(hw, sub == 0 ? a.hc_prep_attn[size_t(l)] : a.hc_prep_ffn[size_t(l)], s)) {
                    return fail(err, "prompt: the HC prepared scales");
                }
            }
            const bool gdn = l % fn::full_attn_interval != fn::full_attn_interval - 1;
            float *    rt  = a.rt_s[size_t(l)];
            if ((gdn && (!fn::dense::pd_bf16_scales(w.alpha, 24, D, a.ab_s[size_t(l)], s) ||
                         !fn::dense::pd_bf16_scales(w.beta, 24, D, a.ab_s[size_t(l)] + 24, s))) ||
                (!gdn && c == w.index_owner &&
                 (!fn::dense::pd_bf16_scales(w.idx_q, 512, D, a.idx_s[size_t(l)], s) ||
                  !fn::dense::pd_bf16_scales(w.idx_k, 128, D, a.idx_s[size_t(l)] + 512, s))) ||
                !fn::dense::pd_bf16_scales(w.router, NE, D, rt, s) ||
                !fn::dense::pd_bf16_scales(w.sh_gate_w, 1, D, rt + NE, s) ||
                !cuda_ok(cudaMemsetAsync(rt + NE + 1, 0, size_t(7) * 4, s), "router pad", err)) {
                return err.empty() ? fail(err, "prompt: the prepared column scales") : false;
            }
        }
        if (d_[c] != nullptr && d_[c]->sparse && !fn::dense::pd_bf16_scales(d_[c]->iw.k_proj, 128, D, a.dik_s, s)) {
            return fail(err, "prompt: the draft index's prepared scales");
        }
        if (!cuda_ok(cudaStreamSynchronize(s), "prepared data", err)) {
            return false;
        }
    }
    // APF2: the startup route report, once a process: each product's route at every rung and every candidate's
    // reason (the table's medians and the cost model's filter and rank)
    static std::atomic<bool> reported{ false };
    if (!reported.exchange(true)) {
        if (!cuda_ok(cudaSetDevice(m_->device[0]), "device", err)) {
            return false;
        }
        std::string r = "ladder: the table holds " + std::to_string(fn::model::lad_table().size()) + " entries\n";
        if (!lad_on_) {
            r += "ladder: the table is UD-Q4_K_XL's; this file runs the retained routes at every rung\n";
        } else {
            for (const fn::model::lad_key & key : prompt_ladder_keys(*p_[0], d_[0], m_->win, 0)) {
                r += fn::model::lad_report(key, lad_model_of(key));
            }
        }
        fprintf(stderr, "%s", r.c_str());
    }
    return true;
}

bool prompt_path::fill_replicas(std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        const int32_t pc = 1 - c, dc = m_->device[c], dp = m_->device[pc];
        card_arena &  a    = a_[size_t(c)];
        const auto    peer = [&](const void * dst, const void * src, int64_t n, const char * what) {
            return cuda_ok(cudaMemcpyPeer(const_cast<void *>(dst), dc, src, dp, size_t(n)), what, err);
        };
        // ME1, one device: the HC and PLE pointers are the peer's resident tensors (init); the draft's norms only
        for (int32_t l = 0; l < fn::n_layer && !m_->single; ++l) {
            if (!m_->win.layer(l)) {
                continue;  // TTTS1: the other box's layer
            }
            for (int32_t sub = 0; sub < 2; ++sub) {
                const card_arena::hc_halves & hh  = sub == 0 ? a.hc_attn[size_t(l)] : a.hc_ffn[size_t(l)];
                const hc_module &             src = sub == 0 ? p_[pc]->w[l].hc_attn : p_[pc]->w[l].hc_ffn;
                if (!peer(hh.down[pc], hc_down_of(src), hc_down_bytes_of(src), "HC down replica") ||
                    !peer(hh.up[pc], hc_up_of(src), hc_up_bytes_of(src), "HC up replica") ||
                    !peer(hh.gamma[pc], src.gamma, hc_gamma_bytes, "HC gamma replica")) {
                    return false;
                }
            }
        }
        const fn::ple::weights & pw = p_[pc]->ple_w;
        if (!m_->single && m_->win.ple() &&
            (!peer(a.ple_key[pc], pw.key, ple_key_bytes_of(pw.kv), "PLE key replica") ||
             !peer(a.ple_value[pc], pw.value, ple_value_bytes_of(pw.kv), "PLE value replica") ||
             !peer(a.ple_nk[pc], pw.norm_key, ple_norm_bytes, "PLE norm replica") ||
             !peer(a.ple_nq[pc], pw.norm_query, ple_norm_bytes, "PLE norm replica") ||
             !peer(a.ple_nc[pc], pw.norm_conv, ple_norm_bytes, "PLE norm replica") ||
             !peer(a.ple_conv[pc], pw.conv, ple_conv_bytes_of(pw.conv32 != nullptr), "PLE conv replica"))) {
            return false;
        }
        if (d_[c] != nullptr) {
            // the draft's full norms: card 0's half first
            for (int32_t h = 0; h < 2; ++h) {
                const draft_card & dd  = *d_[h];
                const int32_t      dev = m_->device[h];
                if (!cuda_ok(cudaMemcpyPeer(a.enorm + ptrdiff_t(h) * F, dc, dd.w.enorm, dev, size_t(F) * 4), "enorm",
                             err)) {
                    return false;
                }
                for (int32_t s = 0; s < 4; ++s) {
                    if (!cuda_ok(cudaMemcpyPeer(a.hnorm + ptrdiff_t(s) * D + ptrdiff_t(h) * F, dc,
                                                dd.w.hnorm + ptrdiff_t(s) * F, dev, size_t(F) * 4),
                                 "hnorm", err)) {
                        return false;
                    }
                }
            }
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
            !cuda_ok(cudaDeviceSynchronize(), "replicas", err)) {
            return false;
        }
    }
    return true;
}

const float * prompt_path::streams_row(int32_t card, int32_t r) const {
    return a_[size_t(card)].S + int64_t(r) * 4 * D;
}

const float * prompt_path::pred_row(int32_t card) const {
    return a_[size_t(card)].pred;
}

bool prompt_path::hold(bool on, std::string & err, const std::vector<byte_span> * need) {
    expert_service * x = p_[0]->c.ex_service;
    if (x == nullptr) {
        return true;
    }
    // ME3: the service takes its loan here with nothing lent too (the zone's staging, sized to the prompt)
    void * const s[2] = { p_[0]->c.stream, p_[1]->c.stream };
    if (!x->prompt_hold(on, s, need, err)) {
        return false;
    }
    if (!x->prompt_lent()) {
        return true;
    }
    held_       = on;
    held_whole_ = on && need == nullptr;
    for (int32_t c = 0; c < 2 && on; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
            !cuda_ok(cudaMemsetAsync(a_[size_t(c)].err, 0, 64, static_cast<cudaStream_t>(s[c])), "error words", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(s[c])), "error words", err)) {
            return false;
        }
    }
    return true;
}

bool prompt_path::test_poison(std::string & err) {
    // every arena byte 0xFF (an F32, FP16 and BF16 NaN), then the error words cleared; the replicas stay
    if (!hold(true, err)) {
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        const card_arena & a = a_[size_t(c)];
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err) ||
            !cuda_ok(cudaMemset(a.base, 0xff, size_t(prompt_arena_bytes(rc_max_, rh_max_, m_->single, c))), "poison",
                     err) ||
            !cuda_ok(cudaMemset(a.err, 0, 64), "poison", err) || !cuda_ok(cudaDeviceSynchronize(), "poison", err)) {
            return false;
        }
    }
    return true;
}

bool prompt_path::run_start(uint32_t cur, std::string & err) {
    // Both cards are drained. Every copy here is ordered on the card's compute stream, which the prompt kernels use:
    // the compute streams are non-blocking, so a copy on the legacy default stream (cudaMemcpyPeer, a device-to-device
    // cudaMemcpy2D) is not ordered before them and a kernel could read the staging buffer's previous contents (PB1's
    // model-load nondeterminism: the peer's PLE history from an earlier run)
    for (int32_t c = 0; c < (m_->single ? 1 : 2); ++c) {
        const int32_t pc = 1 - c;
        card_arena & a = a_[size_t(c)];
        const cudaStream_t s = static_cast<cudaStream_t>(p_[c]->c.stream);
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "device", err)) {
            return false;
        }
        if (d_[c] != nullptr) {
            // the predecessor: both cards' retained carries [4][1280] into [4][2560], card 0's half first
            for (int32_t h = 0; h < 2; ++h) {
                if (!cuda_ok(cudaMemcpy2DAsync(a.pred + ptrdiff_t(h) * F, size_t(D) * 4, d_[h]->carry_prev,
                                               size_t(F) * 4, size_t(F) * 4, 4, cudaMemcpyDefault, s),
                             "predecessor", err)) {
                    return false;
                }
            }
        } else if (!cuda_ok(cudaMemsetAsync(a.pred, 0, size_t(4) * D * 4, s), "predecessor", err)) {
            return false;
        }
        if (!m_->win.ple()) {
            if (!cuda_ok(cudaStreamSynchronize(s), "run start", err)) {
                return false;
            }
            continue;  // TTTS1: the PLE history is the front's
        }
        const float * peer_hist = p_[pc]->ple_hist + int64_t(fn::round::version_slot(cur, 0)) * ple_hist_card;
        if (!m_->single && !cuda_ok(cudaMemcpyPeerAsync(a.ple_stage, m_->device[c], peer_hist, m_->device[pc],
                                                        size_t(ple_hist_card) * 4, s),
                                    "PLE history", err)) {
            return false;
        }
        fn::ple::prompt_args pa;
        pa.hist_out = a.ple_hist[0];
        pa.state = p_[c]->ple_hist;
        pa.cur = p_[c]->c.cur;
        pa.card = c;
        if (!fn::ple::ple_prompt_hist_in(pa, m_->single ? peer_hist : a.ple_stage, s) ||
            !cuda_ok(cudaStreamSynchronize(s), "run start", err)) {
            return err.empty() ? fail(err, "prompt: PLE history in") : false;
        }
    }
    hist_cur_ = 0;
    return true;
}

namespace {

// fn::experts::fmt to fn::dense::fmt for the shared down (EX1's layout)
fn::dense::fmt dense_of(int32_t ef) {
    const auto f = static_cast<fn::experts::fmt>(ef);
    return f == fn::experts::fmt::q2_0 ? fn::dense::fmt::q2_0 :
           f == fn::experts::fmt::q8_0 ? fn::dense::fmt::q8_0 :
                                         fn::dense::fmt::iq4_nl;
}

// APF2: the ladder keys of a call site's packed matrices ws (the runs of one format, at most three each, as dprod)
void dense_run_keys(std::initializer_list<const dense_weight *> ws,
                    int32_t                                     kk,
                    int32_t                                     lay,
                    std::vector<fn::model::lad_key> &           out) {
    const dense_weight * const * w  = ws.begin();
    const int32_t                nw = static_cast<int32_t>(ws.size());
    for (int32_t i = 0; i < nw;) {
        const int32_t i0 = i;
        int32_t       nn = 0;
        for (int32_t m = 0; i < nw && m < 3 && w[i]->fmt == w[i0]->fmt; ++i, ++m) {
            nn += static_cast<int32_t>(w[i]->n_out);
        }
        out.push_back(prompt_dense_key(w[i0]->fmt, lay, kk, nn));
    }
}

fn::dense::pd_act act1(const uint16_t * x, int64_t ld) {
    fn::dense::pd_act a;
    a.a0      = x;
    a.ld0     = ld;
    a.k_split = static_cast<int32_t>(ld);
    return a;
}

}  // namespace

uint8_t * prompt_path::p_msg(int32_t card, bool send, const prompt_chunk & c, int32_t j) const {
    const card_arena & a = a_[size_t(card)];
    return (send ? a.p_send : a.p_recv) + int64_t(j) * fn::xchg::pmsg_payload_at +
           int64_t(prompt_piece_r0(c, j)) * D * 2;
}

bool prompt_path::chunk(const prompt_chunk & c, std::string & err) {
    return chunk_begin(c, err) && chunk_layers(c, 0, fn::n_layer, err) && chunk_end(c, err);
}

bool prompt_path::chunk_begin(const prompt_chunk & c, std::string & err) {
    if (c.rows < 1 || c.rows > rc_ || c.half[0] < 1 || c.half[0] > rh_ || c.half[1] < 0 || c.half[1] > rh_ ||
        c.half[0] + c.half[1] != c.rows || c.tok == nullptr || !prompt_pieces_ok(c) ||
        (m_->win.ple() && (c.e[0] == nullptr || c.e[1] == nullptr))) {
        return fail(err, "prompt: a chunk outside the layout's " + std::to_string(rc_) + " rows in halves of at most " +
                             std::to_string(rh_));
    }
    if (c.n_runs < 0 || c.n_runs > c.rows ||
        (c.n_runs > 0 && (c.runs == nullptr || c.embd[0] == nullptr || c.embd[1] == nullptr))) {
        return fail(err, "prompt: " + std::to_string(c.n_runs) + " supplied runs for " + std::to_string(c.rows) +
                             " rows, or a run without both cards' rows");
    }
    if ((c.planes[0] == nullptr) != (c.planes[1] == nullptr) || (c.planes[0] == nullptr) != (c.bplanes[0] == nullptr) ||
        (c.bplanes[0] == nullptr) != (c.bplanes[1] == nullptr) || (c.n_runs > 0 && c.planes[0] == nullptr)) {
        return fail(err, "prompt: rope planes and block planes come together on both cards (and with supplied rows)");
    }
    for (int32_t i = 0, end = 0; i < c.n_runs; ++i) {  // VIS1: ascending, disjoint, inside the chunk
        const prompt_chunk::supply_run & u = c.runs[i];
        if (u.n < 1 || u.row0 < end || int64_t(u.row0) + u.n > c.rows) {
            return fail(err, "prompt: supplied run " + std::to_string(i) + " (rows " + std::to_string(u.row0) + " .. " +
                                 std::to_string(int64_t(u.row0) + u.n) + ") outside the chunk's " +
                                 std::to_string(c.rows) + " rows or overlapping the run before");
        }
        end = u.row0 + u.n;
    }
    // APF1 Part 4: the chunk's routes (the whole chunk's rows; prompt_route_for). Graphs on the two-card path without
    // the expert service, the host-backed KV tier or the HC dump (whose host copies a capture cannot hold)
    {
        const bool no_xs = p_[0]->c.ex_service == nullptr && p_[1]->c.ex_service == nullptr;
        const bool graph_ok =
            !m_->single && no_xs && p_[0]->c.kv == nullptr && p_[1]->c.kv == nullptr && dump_.row_stride == 0;
        route_ = prompt_route_for(c.rows, graph_ok, force_);
        if (totals_on.load(std::memory_order_relaxed)) {
            const std::lock_guard<std::mutex> g(totals_mu);
            totals_rungs |= 1u << __builtin_ctz(static_cast<uint32_t>(route_.rung / 64));
        }
    }
    // the tokens to both cards (a pinned slot a chunk parity: the slot two chunks back has been copied, the chunk
    // end's join having passed)
    for (int32_t k = 0; k < 2; ++k) {
        card_arena & a = a_[size_t(k)];
        if (!cuda_ok(cudaSetDevice(m_->device[k]), "device", err)) {
            return false;
        }
        // APF2: the split counters zero (the arena may have been lent since the last chunk)
        if (!cuda_ok(cudaMemsetAsync(a.lws.count, 0, size_t(a.lws.count_len) * 4,
                                     static_cast<cudaStream_t>(p_[k]->c.stream)),
                     "ladder counters", err)) {
            return false;
        }
        if (!m_->single || k == 0) {
            int32_t * th = a.tok_host + (c.index & 1) * RC;
            std::memcpy(th, c.tok, size_t(c.rows) * 4);
            if (!cuda_ok(cudaMemcpyAsync(a.tok, th, size_t(c.rows) * 4, cudaMemcpyHostToDevice,
                                         static_cast<cudaStream_t>(p_[k]->c.stream)),
                         "tokens", err)) {
                return false;
            }
        }
        if (test_fail_ == k) {
            const uint32_t one = 1u << 30;
            if (!cuda_ok(
                    cudaMemcpyAsync(a.err, &one, 4, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(p_[k]->c.stream)),
                    "test fail", err) ||
                !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(p_[k]->c.stream)), "test fail", err)) {
                return false;
            }
        }
    }
    test_fail_ = -1;
    if (!m_->win.layer(0)) {
        return true;  // TTTS1: a back stage takes its streams from the crossing (crossing_in), not the embedding
    }
    // the embedding: each card's half, the message, the peer's half (MX6: on single, phase 0 populates both halves in a_[0])
    for (int32_t phase = 0; phase < (m_->single ? 1 : 3); ++phase) {
        for (int32_t k = 0; k < 2; ++k) {
            card_arena & a = a_[size_t(k)];
            link & lk = links_[lk_emb];
            const uint64_t gen = fn::xchg::pgen(c.seq, 0, 0, 0, fn::xchg::pkind::emb);
            cudaStream_t   s   = static_cast<cudaStream_t>(p_[k]->c.stream);
            if (!cuda_ok(cudaSetDevice(m_->device[k]), "device", err)) {
                return false;
            }
            fn::embed::rows_args ea;
            ea.table   = p_[k]->token_embd;
            ea.grids   = p_[k]->c.dense_grids;
            ea.tok     = a.tok;
            ea.rows    = c.rows;
            ea.card    = k;
            ea.own     = reinterpret_cast<float *>(a.emb_send + fn::xchg::pmsg_payload_at);
            ea.full    = a.E;
            ea.streams = a.S;
            ea.err = a.err;
            ea.q8      = p_[k]->embd_q8;  // TTTS2
            if (phase == 0) {
                if (!m_->single && lk.any_copied[k] &&
                    !cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(lk.copied[k]), 0), "emb reuse", err)) {
                    return false;
                }
                // VIS1: supplied rows over the token rows, before the message is stamped (the peer joins them)
                int64_t done = 0;
                bool    sup  = fn::embed::embed_rows(ea, s);
                for (int32_t i = 0; i < c.n_runs && sup; ++i) {
                    fn::embed::supply_args sa;
                    sa.src     = c.embd[k] + done * fn::embed::n_own;
                    sa.row0    = c.runs[i].row0;
                    sa.n       = c.runs[i].n;
                    sa.rows    = c.rows;
                    sa.card    = k;
                    sa.own     = ea.own;
                    sa.full    = ea.full;
                    sa.streams = ea.streams;
                    sup        = fn::embed::embed_supply(sa, s);
                    done += c.runs[i].n;
                }
                if (!sup || (!m_->single && (!fn::xchg::pstamp(a.emb_send, gen, c.rows, fn::xchg::pkind::emb, s) ||
                                             !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(lk.prod[k]), s),
                                                      "emb produced", err)))) {
                    return err.empty() ? fail(err, "prompt: embed") : false;
                }
                lk.g_prod[k] = gen;
                count(psite::embed);
            } else if (phase == 1) {
                cudaStream_t cs = static_cast<cudaStream_t>(copy_[k]);
                if (lk.g_prod[k] != gen ||
                    !cuda_ok(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(lk.prod[k]), 0), "emb wait", err) ||
                    (lk.any_freed[1 - k] &&
                     !cuda_ok(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(lk.freed[1 - k]), 0), "emb free",
                              err)) ||
                    !cuda_ok(cudaMemcpyPeerAsync(lk.recv[1 - k], m_->device[1 - k], lk.send[k], m_->device[k],
                                                 size_t(fn::xchg::pmsg_bytes(int64_t(c.rows) * F * 4)), cs),
                             "emb copy", err) ||
                    !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(lk.copied[k]), cs), "emb copied", err)) {
                    return err.empty() ? fail(err, "prompt: emb generation") : false;
                }
                lk.g_copied[k]   = gen;
                lk.any_copied[k] = true;
            } else {
                if (lk.g_copied[1 - k] != gen ||
                    !cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(lk.copied[1 - k]), 0), "emb recv", err) ||
                    !fn::xchg::pcheck(a.emb_recv, gen, a.err, s) ||
                    !fn::embed::embed_join(ea, reinterpret_cast<const float *>(a.emb_recv + fn::xchg::pmsg_payload_at),
                                           s) ||
                    !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(lk.freed[k]), s), "emb freed", err)) {
                    return err.empty() ? fail(err, "prompt: embed join") : false;
                }
                lk.g_freed[k]   = gen;
                lk.any_freed[k] = true;
            }
        }
    }
    return true;
}

bool prompt_path::chunk_layers(const prompt_chunk & c, int32_t lo, int32_t hi, std::string & err) {
    if (lo < 0 || hi > fn::n_layer || lo > hi) {
        return fail(err, "prompt: layers " + std::to_string(lo) + " .. " + std::to_string(hi) + " outside 0 .. 48");
    }
    const int32_t np = prompt_pieces(c);
    // the layers, sublayer-major in pieces (MUP2; the halves when the chunk names none)
    for (int32_t l = lo; l < hi; ++l) {
        for (int32_t sub = 0; sub < 2; ++sub) {
            for (int32_t j = 0; j < np; ++j) {
                if (!chunk_piece(c, l, sub, j, err)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool prompt_path::chunk_piece(const prompt_chunk & c, int32_t l, int32_t sub, int32_t j, std::string & err) {
    if (l < 0 || l >= fn::n_layer || sub < 0 || sub > 1 || j < 0 || j >= prompt_pieces(c)) {
        return fail(err, "prompt: a unit outside the chunk (layer " + std::to_string(l) + ", sublayer " +
                             std::to_string(sub) + ", piece " + std::to_string(j) + ")");
    }
    if (route_.graph) {
        return unit_graph(c, l, sub, j, err);
    }
    note(&route_counts::eager_units);
    return unit_eager(c, l, sub, j, err);
}

// APF2: a unit graph's key: the unit (layer, sublayer, piece), its rows' rung, and for piece 0 (whose conversions
// follow every piece's route) the chunk's pieces' rungs; the forced routes. Only the capture count depends on it: a
// replay checks its operations against the graph's (graph_rec) and captures again on any difference
uint64_t prompt_path::unit_key(const prompt_chunk & c, int32_t l, int32_t sub, int32_t j) const {
    uint64_t   h   = 1469598103934665603ull;
    const auto mix = [&](uint64_t v) {
        h = (h ^ v) * 1099511628211ull;
    };
    mix(uint64_t(l));
    mix(uint64_t(sub));
    mix(uint64_t(j));
    mix(static_cast<uint64_t>(fn::model::lad_rung_index(prompt_piece_rows(c, j))) + 1);
    if (j == 0) {
        for (int32_t i = 0; i < prompt_pieces(c); ++i) {
            mix(uint64_t(fn::model::lad_rung_index(prompt_piece_rows(c, i)) + 1) << 8);
        }
    }
    mix(uint64_t(force_.dense) | uint64_t(force_.hc) << 4 | uint64_t(force_.expert) << 8);
    return h;
}

bool prompt_path::unit_eager(const prompt_chunk & c, int32_t l, int32_t sub, int32_t j, std::string & err) {
    for (int32_t phase = 0; phase < 4; ++phase) {
        if (!layer_half(c, l, sub, j, phase, err)) {
            return false;
        }
    }
    return true;
}

// ---- APF1 Part 4: a unit as one CUDA graph --------------------------------------------------------------------------
// The unit's launches (both cards' compute streams and copy streams) captured from card 0's compute stream and
// replayed as one graph: the same kernels with the same arguments in the same stream order, so the same results.
// Before the capture card 0's compute stream waits for the tails of the other three streams (the pre-join), so the
// graph starts after everything enqueued before it and the unit's waits on events recorded earlier are already met
// (ev_wait skips them under the capture; a capture cannot wait on work outside it). The capture forks the other
// streams from card 0's and joins them back. After the replay the other streams wait for it, and the link events the
// unit recorded are recorded again on their card's compute stream, so later units, the chunk's end, publication and
// the other lane see them complete as before (later than the copy itself: never early). The executable is kept per
// unit position and updated in place with the new arguments (cudaGraphExecUpdate), instantiated again when the
// topology differs. A capture that fails (an operation capture refuses) restores the host state it advanced and
// runs the unit eagerly.
bool prompt_path::unit_graph(const prompt_chunk & c, int32_t l, int32_t sub, int32_t j, std::string & err) {
    cudaStream_t st[2] = { static_cast<cudaStream_t>(p_[0]->c.stream), static_cast<cudaStream_t>(p_[1]->c.stream) };
    cudaStream_t cp[2] = { static_cast<cudaStream_t>(copy_[0]), static_cast<cudaStream_t>(copy_[1]) };
    const auto   ev    = [&](int32_t card, int32_t i) {
        return static_cast<cudaEvent_t>(gj_[card][i]);
    };
    const auto dev = [&](int32_t card) {
        return cuda_ok(cudaSetDevice(m_->device[card]), "device", err);
    };
    for (int32_t card = 0; card < 2; ++card) {
        for (int32_t i = 0; i < 4; ++i) {
            if (gj_[card][i] == nullptr) {
                cudaEvent_t e = nullptr;
                if (!dev(card) || !cuda_ok(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "graph event", err)) {
                    return false;
                }
                gj_[card][i] = e;
            }
        }
    }
    // the pre-join: card 0's compute stream after the tails of card 1's compute and both copy streams
    if (!dev(1) || !cuda_ok(cudaEventRecord(ev(1, 0), st[1]), "pre-join", err) ||
        !cuda_ok(cudaEventRecord(ev(1, 1), cp[1]), "pre-join", err) || !dev(0) ||
        !cuda_ok(cudaEventRecord(ev(0, 0), cp[0]), "pre-join", err) ||
        !cuda_ok(cudaStreamWaitEvent(st[0], ev(0, 0), 0), "pre-join", err) ||
        !cuda_ok(cudaStreamWaitEvent(st[0], ev(1, 0), 0), "pre-join", err) ||
        !cuda_ok(cudaStreamWaitEvent(st[0], ev(1, 1), 0), "pre-join", err)) {
        return false;
    }
    const std::vector<link>             links0 = links_;
    const std::array<int64_t, n_psites> sites0 = sites_;
    const route_counts                  rc0    = rcount_;
    // the fork (events 1 of card 0 and the joins' events 2, 3 are recorded only inside the graph), the unit, the join
    const auto                          body   = [&]() {
        return cuda_ok(cudaEventRecord(ev(0, 1), st[0]), "fork", err) &&
               cuda_ok(cudaStreamWaitEvent(cp[0], ev(0, 1), 0), "fork", err) && dev(1) &&
               cuda_ok(cudaStreamWaitEvent(st[1], ev(0, 1), 0), "fork", err) &&
               cuda_ok(cudaStreamWaitEvent(cp[1], ev(0, 1), 0), "fork", err) && unit_eager(c, l, sub, j, err) &&
               dev(1) && cuda_ok(cudaEventRecord(ev(1, 2), st[1]), "join", err) &&
               cuda_ok(cudaEventRecord(ev(1, 3), cp[1]), "join", err) && dev(0) &&
               cuda_ok(cudaEventRecord(ev(0, 2), cp[0]), "join", err) &&
               cuda_ok(cudaStreamWaitEvent(st[0], ev(1, 2), 0), "join", err) &&
               cuda_ok(cudaStreamWaitEvent(st[0], ev(1, 3), 0), "join", err) &&
               cuda_ok(cudaStreamWaitEvent(st[0], ev(0, 2), 0), "join", err);
    };
    const auto restore = [&]() {
        links_ = links0;
        sites_ = sites0;
        undo_totals(rcount_, rc0);
        rcount_ = rc0;
        err.clear();
    };
    const uint64_t                    key = unit_key(c, l, sub, j);
    std::shared_ptr<unit_graph_entry> ge;
    if (const auto it = ugraphs_.find(key); it != ugraphs_.end()) {
        ge = it->second;
    }
    cudaGraphExec_t x = nullptr;
    // APF2: the unit's graph replayed: the host code runs with every operation's arguments going into the executable
    // (graph_rec's update; no stream capture). A mismatch (another sequence of operations) captures again
    if (ge != nullptr && ge->updatable) {
        cap_rec_[0].clear();
        cap_rec_[1].clear();
        graph_rec rec;
        rec.begin_update(&ge->ops, ge->exec);
        capturing_       = true;
        const bool ok    = body();
        capturing_       = false;
        const bool match = rec.end();
        if (ok && match) {
            x = ge->exec;
            note(&route_counts::graph_updated);
        } else {
            restore();
            ugraphs_.erase(key);
            ge = nullptr;
        }
    }
    if (x == nullptr) {
        cap_rec_[0].clear();
        cap_rec_[1].clear();
        if (!dev(0) || !cuda_ok(cudaStreamBeginCapture(st[0], cudaStreamCaptureModeRelaxed), "capture", err)) {
            return false;
        }
        auto      ne = std::make_shared<unit_graph_entry>();
        graph_rec rec;
        rec.begin_capture(&ne->ops);
        capturing_           = true;
        const bool ok        = body();
        capturing_           = false;
        ne->updatable        = rec.end();
        cudaGraph_t       g  = nullptr;
        const cudaError_t ce = (cudaSetDevice(m_->device[0]), cudaStreamEndCapture(st[0], &g));
        if (!ok || ce != cudaSuccess || g == nullptr) {
            // the unit was not run: the host state back, then eagerly (a failure there is the unit's own)
            if (g != nullptr) {
                cudaGraphDestroy(g);
            }
            (void) cudaGetLastError();
            fprintf(stderr,
                    "prompt: layer %d sublayer %d piece %d: graph capture failed (%s%s%s); the unit runs eagerly\n", l,
                    sub, j, cudaGetErrorString(ce), ok ? "" : "; ", ok ? "" : err.c_str());
            restore();
            note(&route_counts::eager_units);
            return unit_eager(c, l, sub, j, err);
        }
        ne->graph    = g;
        size_t nodes = 0;
        if (!cuda_ok(cudaGraphGetNodes(g, nullptr, &nodes), "graph nodes", err) ||
            !cuda_ok(cudaGraphInstantiate(&ne->exec, g, 0), "graph instantiate", err)) {
            return false;
        }
        graph_nodes_ += static_cast<int64_t>(nodes);
        note(&route_counts::graph_captures);
        x             = ne->exec;
        ugraphs_[key] = ne;
    }
    if (!cuda_ok(cudaGraphLaunch(x, st[0]), "graph launch", err)) {
        return false;
    }
    // after the replay: the other streams wait for it; the unit's link events recorded again on their card
    if (!cuda_ok(cudaEventRecord(ev(0, 3), st[0]), "done", err) ||
        !cuda_ok(cudaStreamWaitEvent(cp[0], ev(0, 3), 0), "done", err) || !dev(1) ||
        !cuda_ok(cudaStreamWaitEvent(st[1], ev(0, 3), 0), "done", err) ||
        !cuda_ok(cudaStreamWaitEvent(cp[1], ev(0, 3), 0), "done", err)) {
        return false;
    }
    for (int32_t card = 0; card < 2; ++card) {
        if (!dev(card)) {
            return false;
        }
        for (void * e : cap_rec_[card]) {
            if (!cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(e), st[card]), "link event", err)) {
                return false;
            }
        }
    }
    note(&route_counts::graph_units);
    return true;
}

bool prompt_path::ev_wait(void * stream, void * event, std::string & err, const char * what) {
    if (capturing_) {
        bool in = false;
        for (const std::vector<void *> & v : cap_rec_) {
            in = in || std::find(v.begin(), v.end(), event) != v.end();
        }
        if (!in) {
            return true;  // recorded before the capture: the pre-join already waited for it
        }
    }
    return cuda_ok(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), static_cast<cudaEvent_t>(event), 0), what,
                   err);
}

bool prompt_path::ev_record(void * event, int32_t card, void * stream, std::string & err, const char * what) {
    if (capturing_ && std::find(cap_rec_[card].begin(), cap_rec_[card].end(), event) == cap_rec_[card].end()) {
        cap_rec_[card].push_back(event);
    }
    return cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)), what, err);
}

bool prompt_path::peer_copy(void *        dst,
                            int32_t       dst_card,
                            const void *  src,
                            int32_t       src_card,
                            size_t        bytes,
                            void *        stream,
                            std::string & err) {
    // a capture refuses cudaMemcpyPeerAsync; the same copy by its unified addresses (a peer copy node)
    if (capturing_) {
        return cuda_ok(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, static_cast<cudaStream_t>(stream)), "copy",
                       err);
    }
    return cuda_ok(cudaMemcpyPeerAsync(dst, m_->device[dst_card], src, m_->device[src_card], bytes,
                                       static_cast<cudaStream_t>(stream)),
                   "copy", err);
}

bool prompt_path::chunk_end(const prompt_chunk & c, std::string & err) {
    const int32_t nh = c.half[1] > 0 ? 2 : 1;
    // the target's end: layer 47's FFN exchange into the final streams (phase 0 of a layer 48 attention sublayer),
    // piece by piece (MUP2); TTTS1: the window's last layer's (split-test's one-box windows end before 48)
    for (int32_t j = 0; j < prompt_pieces(c); ++j) {
        if (!layer_half(c, m_->win.hi + 1, 0, j, 0, err)) {
            return false;
        }
    }
    // the draft's catch-up stays on the halves (a partition refines them)
    if (c.draft && d_[0] != nullptr) {
        for (int32_t h = 0; h < nh; ++h) {
            for (int32_t phase = 0; phase < 5; ++phase) {
                if (!draft_half(c, h, phase, err)) {
                    return false;
                }
            }
        }
    }
    return true;
}

// One phase of sublayer sub of layer l for piece h (MUP2: the halves generalized; prompt_pieces), both cards (card 0,
// then card 1):
//   0: the previous sublayer's exchange of this half (receive, combine into S); at layer 1's attention, PLE; HC; the
//      body's first part (products; on a QSA layer's index owner the index and selection, the SEL message produced)
//   1: the SEL copy (QSA layers, the owner)
//   2: the body's rest (QSA: the peer acquires SEL; K/V; attention; wo), the partial packed, the P message produced
//   3: the P copy
// l = 48 runs phase 0's receive and combine only (the target's end).
std::vector<fn::model::lad_key> prompt_ladder_keys(const pass_card &  p,
                                                   const draft_card * d,
                                                   const window &     win,
                                                   int32_t            card) {
    std::vector<fn::model::lad_key> k;
    const auto                      bf = [](int32_t kk, int32_t nn) {
        return prompt_dense_key(static_cast<int32_t>(fn::dense::fmt::bf16), 0, kk, nn);
    };
    for (int32_t l = 0; l < fn::n_layer; ++l) {
        if (!win.layer(l)) {
            continue;
        }
        const layer_weights & w   = p.w[l];
        const bool            gdn = l % fn::full_attn_interval != fn::full_attn_interval - 1;
        k.push_back(prompt_hc_key(false, w.hc_attn.q8_down != nullptr));
        k.push_back(prompt_hc_key(true, w.hc_attn.q8_down != nullptr));
        if (gdn) {
            dense_run_keys({ &w.qkv, &w.z }, D, 0, k);
            k.push_back(bf(D, 48));
            dense_run_keys({ &w.ssm_out }, GV, 0, k);
        } else {
            dense_run_keys({ &w.wq, &w.wk, &w.wv }, D, 0, k);
            k.push_back(prompt_dense_key(w.wo.fmt, 0, GV, D));
            if (card == w.index_owner) {
                k.push_back(bf(D, 640));
            }
        }
        k.push_back(prompt_hc_key(false, w.hc_ffn.q8_down != nullptr));
        k.push_back(prompt_hc_key(true, w.hc_ffn.q8_down != nullptr));
        k.push_back(bf(D, NE + 8));
        dense_run_keys({ &w.sh_gate, &w.sh_up }, D, 0, k);
        dense_weight sd;
        sd.fmt   = static_cast<int32_t>(dense_of(w.ex_sh_down_fmt));
        sd.n_out = D;
        dense_run_keys({ &sd }, 320, static_cast<int32_t>(fn::dense::pd_layout::ex1_down), k);
        if (fn::experts::el_takes_gate_up(static_cast<fn::experts::fmt>(w.ex_gu_fmt))) {
            k.push_back(prompt_expert_key(false, w.ex_gu_fmt));
        }
        if (fn::experts::el_takes_down(static_cast<fn::experts::fmt>(w.ex_down_fmt))) {
            k.push_back(prompt_expert_key(true, w.ex_down_fmt));
        }
    }
    if (d != nullptr && win.layer(fn::n_layer)) {
        k.push_back(prompt_dense_key(d->w.eh.fmt, 0, 2 * D, F));
        k.push_back(prompt_dense_key(d->w.wk.fmt, 0, D, 256));
        if (d->sparse) {
            k.push_back(bf(D, 128));
        }
    }
    std::vector<fn::model::lad_key> u;
    for (const fn::model::lad_key & x : k) {
        if (std::find(u.begin(), u.end(), x) == u.end()) {
            u.push_back(x);
        }
    }
    return u;
}

bool prompt_path::layer_half(const prompt_chunk & c,
                             int32_t              l,
                             int32_t              sub,
                             int32_t              h,
                             int32_t              phase,
                             std::string &        err) {
    const int32_t  r0       = prompt_piece_r0(c, h);
    const int32_t  n        = prompt_piece_rows(c, h);
    const int64_t  pos0     = c.start + r0;
    const bool     end      = l == m_->win.hi + 1;  // the window's end (TTTS1; the whole model: 48)
    const bool     gdn      = !end && l % fn::full_attn_interval != fn::full_attn_interval - 1;
    link &         lp       = links_[lk_p0 + h];
    // SEL: produced and consumed inside the piece's own sublayer, so two slots alternate (their reuse waits)
    link &         ls       = links_[h % 2 == 0 ? lk_sel0 : lk_sel1];
    const int32_t  sh       = h % 2;  // the SEL slot
    // the previous sublayer of this half: (l, sub - 1) or (l - 1, 1)
    const bool     first    = l == 0 && sub == 0;
    const uint64_t gen_prev = sub == 1 ? fn::xchg::pgen(c.seq, l, 0, h, fn::xchg::pkind::attn) :
                                         fn::xchg::pgen(c.seq, l - 1, 1, h, fn::xchg::pkind::ffn);
    const uint64_t gen      = fn::xchg::pgen(c.seq, l, sub, h, sub == 0 ? fn::xchg::pkind::attn : fn::xchg::pkind::ffn);
    const uint64_t gen_sel  = fn::xchg::pgen(c.seq, l, 0, h, fn::xchg::pkind::sel);
    if (phase == 1 || phase == 3) {
        if (end || (phase == 1 && (sub != 0 || gdn))) {
            return true;
        }
        link &         lk    = phase == 1 ? ls : lp;
        const uint64_t g     = phase == 1 ? gen_sel : gen;
        const int64_t  bytes = phase == 1 ? sel_slot : fn::xchg::pmsg_bytes(int64_t(n) * D * 2);
        // the SEL payload as qsp_select writes it: the header and the half's slabs' bodies (slab k at k x prompt_slab
        // x max_row_words words), then the slabs' offsets at prompt_body_words words ([slabs][prompt_slab + 1] I32)
        const int64_t  n_slab        = (n + fn::qsa::prompt_slab - 1) / fn::qsa::prompt_slab;
        const int64_t  sel_used      = fn::xchg::pmsg_payload_at + n_slab * fn::qsa::prompt_slab_words * 4;
        const int64_t  sel_offs_at   = fn::xchg::pmsg_payload_at + fn::qsa::prompt_body_words * 4;
        const int64_t  sel_offs_used = n_slab * (fn::qsa::prompt_slab + 1) * 4;
        for (int32_t k = 0; k < 2; ++k) {
            if (phase == 1 && k != p_[0]->w[l].index_owner) {
                continue;
            }
            if (lk.g_prod[k] != g) {
                return fail(err, "prompt: a message copied before its generation was produced");
            }
            if (!m_->single) {
                cudaStream_t    cs  = static_cast<cudaStream_t>(copy_[k]);
                uint8_t * const dst = phase == 1 ? lk.recv[1 - k] : p_msg(1 - k, false, c, h);
                uint8_t * const src = phase == 1 ? lk.send[k] : p_msg(k, true, c, h);
                if (!cuda_ok(cudaSetDevice(m_->device[k]), "device", err) ||
                    !ev_wait(cs, lk.prod[k], err, "copy wait") ||
                    (lk.any_freed[1 - k] && !ev_wait(cs, lk.freed[1 - k], err, "copy free")) ||
                    !peer_copy(dst, 1 - k, src, k, size_t(phase == 1 ? sel_used : bytes), cs, err) ||
                    // SEL (PB3): the used slabs' bodies above, then their offsets (the payload's two regions)
                    (phase == 1 &&
                     !peer_copy(dst + sel_offs_at, 1 - k, src + sel_offs_at, k, size_t(sel_offs_used), cs, err)) ||
                    !ev_record(lk.copied[k], k, cs, err, "copied")) {
                    return false;
                }
            }
            lk.g_copied[k] = g;
            lk.any_copied[k] = true;
        }
        return true;
    }
    // ME4: a ring layer's expert launches, deferred from each logical card's FFN body to the group loop after both
    struct ring_card {
        bool                         on = false;
        fn::experts::fmt             gf = fn::experts::fmt::q2_0, df = fn::experts::fmt::q2_0;
        fn::experts::ep_gate_up_args gu;
        fn::experts::ep_down_args    dn;
        fn::experts::ep_sum_args     sa;
        int32_t * toff = nullptr;
    };
    ring_card rcard[2];
    // MX7: the CPU share's job joined on every exit of the layer once begun (prompt_share_settle)
    struct share_settle {
        expert_service * xs = nullptr;
        int32_t l = -1;
        ~share_settle() {
            if (xs != nullptr) {
                xs->prompt_share_settle(l);
            }
        }
    } settle;
    settle.l = l;
    for (int32_t k = 0; k < 2; ++k) {
        card_arena & a = a_[size_t(k)];
        pass_card & p = *p_[k];
        cudaStream_t s = static_cast<cudaStream_t>(p.c.stream);
        if (!cuda_ok(cudaSetDevice(m_->device[k]), "device", err)) {
            return false;
        }
        float * S   = a.S + int64_t(r0) * 4 * D;
        float * inj = a.inject + int64_t(r0) * 4;
        if (phase == 0) {
            // 0a: the previous sublayer's exchange. PB3: folded into HC's prescale (hct_mix reads the streams once),
            // except at the end and before layer 1's PLE (which changes the streams between them); the slots are
            // freed after their reader
            const bool fold = !first && !end && !(l == 1 && sub == 0);
            fn::hc::hct_combine cb;
            const auto          freed = [&]() {
                if (!m_->single && !ev_record(lp.freed[k], k, s, err, "freed")) {
                    return false;
                }
                lp.g_freed[k] = gen_prev;
                lp.any_freed[k] = true;
                return true;
            };
            if (!first && (!m_->single || k == 0)) {
                if (lp.g_copied[1 - k] != gen_prev) {
                    return fail(err, "prompt: a partial received before its generation was copied");
                }
                if (!m_->single && !ev_wait(s, lp.copied[1 - k], err, "recv")) {
                    return false;
                }
                if (fold) {
                    cb.card    = k;
                    cb.own     = p_msg(k, true, c, h);
                    cb.peer    = p_msg(k, false, c, h);
                    cb.gen     = gen_prev;
                    cb.inject  = inj;
                    cb.streams = S;
                } else if (!fn::xchg::pcombine(n, k, p_msg(k, true, c, h), p_msg(k, false, c, h), gen_prev, inj, S,
                                               a.err, s) ||
                           !freed()) {
                    return err.empty() ? fail(err, "prompt: combine") : false;
                }
                count(psite::exchange);
            }
            if (end) {
                continue;
            }
            const layer_weights & w  = p.w[l];
            const layer_state & st = p.s[l];
            uint8_t * sc = a.scratch;
            // 0b: PLE before layer 1's first HC
            if (l == 1 && sub == 0 && (!m_->single || k == 0)) {
                bump b{sc};
                float * key = b.take<float>(int64_t(n) * 4 * D * 4);
                float * x =
                    m_->single ? reinterpret_cast<float *>(a_[1].scratch) : b.take<float>(int64_t(n) * 4 * D * 4);
                float * value = b.take<float>(int64_t(n) * D * 4);
                float * scal = b.take<float>(int64_t(n) * 8 * 4);
                fn::dense::pb_args kb;
                kb.w_lo               = a.ple_key[0];
                kb.w_hi               = a.ple_key[1];
                kb.period             = D;
                kb.half               = F;
                kb.n                  = 4 * D;
                kb.k                  = D;
                kb.rows               = n;
                kb.x                  = c.e[k] + int64_t(r0) * D;
                kb.ldx                = D;
                kb.out                = key;
                kb.ld_out             = int64_t(4) * D;
                kb.f32                = p.ple_w.kv == fn::ple::kv_fmt::f32;   // TTTS2
                kb.q8                 = p.ple_w.kv == fn::ple::kv_fmt::q8_0;  // TTTS3
                fn::dense::pb_args vb = kb;
                vb.w_lo               = a.ple_value[0];
                vb.w_hi               = a.ple_value[1];
                vb.n                  = D;
                vb.out                = value;
                vb.ld_out             = D;
                fn::ple::prompt_args pa;
                for (int32_t hh = 0; hh < 2; ++hh) {
                    pa.w.norm_key[hh]   = a.ple_nk[hh];
                    pa.w.norm_query[hh] = a.ple_nq[hh];
                    pa.w.norm_conv[hh]  = a.ple_nc[hh];
                    pa.w.conv[hh]       = a.ple_conv[hh];
                }
                pa.w.conv32 = p.ple_w.conv32 != nullptr;  // TTTS2
                pa.rows     = n;
                pa.key      = key;
                pa.value    = value;
                pa.streams  = S;
                pa.scal     = scal;
                pa.x        = x;
                pa.hist_in  = a.ple_hist[hist_cur_];
                pa.hist_out = a.ple_hist[1 - hist_cur_];
                pa.state    = p.ple_hist;
                pa.cur      = p.c.cur;
                pa.v_out    = prompt_piece_v_out(h);
                pa.card     = k;
                if (!fn::dense::pd_bf16(kb, s) || !fn::dense::pd_bf16(vb, s) || !fn::ple::ple_prompt_stats(pa, s) ||
                    !fn::ple::ple_prompt_x(pa, s) || !fn::ple::ple_prompt_update(pa, s) ||
                    !fn::ple::ple_prompt_hist(pa, s)) {
                    return fail(err, "prompt: PLE");
                }
                if (m_->single) {
                    fn::ple::prompt_args pa1 = pa;
                    pa1.state = p_[1]->ple_hist;
                    pa1.cur = p_[1]->c.cur;
                    pa1.card = 1;
                    if (!fn::ple::ple_prompt_hist(pa1, s)) {
                        return fail(err, "prompt: PLE");
                    }
                }
                count(psite::ple);
                note(&route_counts::ple);
            }
            // 0c: HC
            if (!m_->single || k == 0) {
                const card_arena::hc_halves & hh = sub == 0 ? a.hc_attn[size_t(l)] : a.hc_ffn[size_t(l)];
                bump b{sc};
                fn::hc::hcp_args ha;
                ha.w.form = fn::hc::hcp_form::replicated;
                ha.w.q8   = (sub == 0 ? w.hc_attn : w.hc_ffn).q8_down != nullptr;  // TTTS2
                for (int32_t q = 0; q < 2; ++q) {
                    ha.w.down[q]  = hh.down[q];
                    ha.w.up[q]    = hh.up[q];
                    ha.w.gamma[q] = hh.gamma[q];
                }
                ha.rows       = n;
                ha.x          = S;
                ha.mom        = b.take<float>(int64_t(n) * fn::hc::hcp_mom * 4);
                ha.low        = b.take<float>(int64_t(n) * 320 * 4);
                ha.rinv       = b.take<float>(int64_t(n) * 4 * 4);
                ha.inject     = inj;
                // PB3: the F32 copy only for the HC dumps (the body reads mixed16; the shared gate is in the router's
                // product)
                const bool dumping = dump_.row_stride > 0 &&
                                     std::find(dump_.layers.begin(), dump_.layers.end(), l) != dump_.layers.end() &&
                                     std::find(dump_.chunks.begin(), dump_.chunks.end(), c.index) != dump_.chunks.end();
                ha.mixed = dumping ? a.mixed : nullptr;
                ha.ld_mixed   = D;
                ha.mixed16    = a.mixed16;
                ha.ld_mixed16 = D;
                ha.err        = a.err;
                ha.card       = k;
                // APF2: both products on the ladder's tiles (the table's or the force's routes at the piece's rows;
                // split K in the card's workspace), else the retained route: the coefficients to FP16 at the
                // sublayer's first piece when any piece of the chunk takes it (its other pieces read them)
                uint8_t * const             hs = b.take<uint8_t>(fn::hc::hct_scratch_bytes(n));
                const fn::hc::hct_weights & hp = sub == 0 ? a.hc_prep_attn[size_t(l)] : a.hc_prep_ffn[size_t(l)];
                const fn::model::lad_key    kd = prompt_hc_key(false, ha.w.q8), ku = prompt_hc_key(true, ha.w.q8);
                const fn::model::lad_cfg    cd = lad_route(kd, n, force_.hc), cu = lad_route(ku, n, force_.hc);
                const bool                  hz = !cd.retained() && !cu.retained();
                const bool hconv = h == 0 && (lad_retained_any(kd, c, force_.hc) || lad_retained_any(ku, c, force_.hc));
                fn::hc::hct_route hr;
                hr.down_tile   = cd.tile;
                hr.down_splits = cd.splits;
                hr.up_tile     = cu.tile;
                hr.up_splits   = cu.splits;
                if ((hconv && !fn::hc::hct_convert(ha.w, a.hct, s)) ||
                    (hz ? !fn::hc::hct_mix_ladder(ha, cb, hp, hs, hr, a.lws, s) :
                          !fn::hc::hct_mix(ha, cb, a.hct, hs, s)) ||
                    (fold && !freed())) {
                    return err.empty() ? fail(err, "prompt: HC") : false;
                }
                note(hz ? &route_counts::hc_ladder : &route_counts::hc_retained);
                count(sub == 0 ? psite::hc_attn : psite::hc_ffn);
                if (k == 0 && (!dump_hc(c, l, sub, h, err) || !dump_hc(c, l, sub + 2, h, err))) {
                    return false;
                }
            }
            // 0d: the body's first part. PB3: the sublayer's producer (GDN out, the experts' sum, QSA wo) writes the P
            // message's FP16 payload itself, so the send slot's previous message has been copied out before the body
            if (!m_->single && lp.any_copied[k] && !ev_wait(s, lp.copied[k], err, "slot reuse")) {
                return false;
            }
            uint16_t * const pay = reinterpret_cast<uint16_t *>(p_msg(k, true, c, h) + fn::xchg::pmsg_payload_at);
            const fn::dense::pd_act mx   = act1(a.mixed16, D);
            // PB3: the sublayer's dense weights to FP16 in the conversion region at its first piece (fused matrices
            // contiguous), then tensor products with one output segment a matrix. APF2: each product's route from
            // the ladder at the piece's rows (lad_route): a ladder tile decodes its weights in the tile loader
            // (pl_tc), the retained route reads the conversion, made at piece 0 when any piece of the chunk takes it
            const auto              to16 = [&](const dense_weight & dw, int64_t at, bool conv,
                                               fn::dense::pd_layout lay = fn::dense::pd_layout::dp2) {
                fn::dense::pd_args pa;
                pa.f     = static_cast<fn::dense::fmt>(dw.fmt);
                pa.lay   = lay;
                pa.k     = static_cast<int32_t>(dw.k);
                pa.n     = static_cast<int32_t>(dw.n_out);
                pa.w     = dw.w;
                pa.grids = p.c.dense_grids;
                return !conv || fn::dense::pd_convert(pa, a.w16 + at, s);
            };
            // whether the sublayer's piece 0 converts key's weights (some piece of the chunk takes the retained route)
            const auto dconv = [&](const fn::model::lad_key & key, int32_t mul = 1) {
                return h == 0 && lad_retained_any(key, c, force_.dense, mul);
            };
            // a ladder product: pl_tc on the route's tile and split count, noted
            const auto ltc = [&](const fn::dense::pq_args & q, const fn::model::lad_cfg & cfg) {
                note(&route_counts::dense_ladder);
                if (cfg.splits > 1) {
                    note(&route_counts::dense_split);
                }
                note_fmt(&route_counts::dense_fmts, static_cast<int32_t>(q.n_src == 0 ? fn::dense::fmt::bf16 : q.f));
                return fn::dense::pl_tc(q, cfg.tile, cfg.splits, a.lws, s);
            };
            // segs: up to three output segments; cs: the BF16 conversions' column scales (at cs_at), or -1; o16: the
            // P payload instead of the one segment
            const auto tprod = [&](int64_t at, int32_t nn, int32_t kk, const fn::dense::pd_act & x,
                                   const fn::dense::pt_seg * segs, int32_t n_seg, int64_t cs_at,
                                   uint16_t * o16 = nullptr, int64_t l16 = D, uint32_t e16 = fn::xchg::err_prange,
                                   const float * row_scale = nullptr) {
                fn::dense::pt_args t;
                t.w = a.w16 + at;
                t.n = nn;
                t.k = kk;
                t.rows = n;
                t.x = x;
                t.n_seg = n_seg;
                for (int32_t i = 0; i < n_seg; ++i) {
                    t.seg[i] = segs[i];
                }
                t.col_scale = cs_at < 0 ? nullptr : reinterpret_cast<const float *>(a.w16 + cs_at);
                t.row_scale = row_scale;
                t.out16 = o16;
                t.ld16 = l16;
                t.err = a.err;
                t.err_bit = e16;
                note(&route_counts::dense_retained);
                return fn::dense::pd_tc(t, s);
            };
            // one product of packed matrices ws (columns one after another, K kk; converted at at). APF2: the runs
            // of one format (at most three sources each, pl_tc's), each its own key and route; the product takes the
            // ladder when every run's route is a ladder tile, else the retained route (one pd_tc over the conversion)
            const auto dprod = [&](std::initializer_list<const dense_weight *> ws, int64_t at, int32_t kk,
                                   const fn::dense::pd_act & x, const fn::dense::pt_seg * segs, int32_t n_seg,
                                   uint16_t * o16 = nullptr, int64_t l16 = D, uint32_t e16 = fn::xchg::err_prange,
                                   fn::dense::pd_layout lay       = fn::dense::pd_layout::dp2,
                                   const float *        row_scale = nullptr) {
                const dense_weight * const * w  = ws.begin();
                const int32_t                nw = static_cast<int32_t>(ws.size());
                int32_t ri[4] = {}, rcol[4] = {}, nr = 0;  // run r: sources ri[r] .., columns rcol[r] ..
                for (int32_t i = 0, c0 = 0; i < nw && nr < 3; ++nr) {
                    ri[nr]   = i;
                    rcol[nr] = c0;
                    for (int32_t m = 0; i < nw && m < 3 && w[i]->fmt == w[ri[nr]]->fmt; ++i, ++m) {
                        c0 += static_cast<int32_t>(w[i]->n_out);
                    }
                    ri[nr + 1]   = i;
                    rcol[nr + 1] = c0;
                }
                fn::model::lad_cfg cfg[3];
                bool               ladder = true, conv = false;
                for (int32_t r = 0; r < nr; ++r) {
                    const fn::model::lad_key key =
                        prompt_dense_key(w[ri[r]]->fmt, static_cast<int32_t>(lay), kk, rcol[r + 1] - rcol[r]);
                    cfg[r] = lad_route(key, n, force_.dense);
                    ladder = ladder && !cfg[r].retained();
                    conv   = conv || dconv(key);
                }
                int32_t nn = 0;
                for (const dense_weight * dw : ws) {
                    if (!to16(*dw, at + int64_t(nn) * kk, conv, lay)) {
                        return false;
                    }
                    nn += static_cast<int32_t>(dw->n_out);
                }
                if (!ladder) {
                    return tprod(at, nn, kk, x, segs, n_seg, -1, o16, l16, e16, row_scale);
                }
                for (int32_t r = 0; r < nr; ++r) {
                    fn::dense::pq_args q;
                    q.f              = static_cast<fn::dense::fmt>(w[ri[r]]->fmt);
                    q.lay            = lay;
                    q.grids          = p.c.dense_grids;
                    const int32_t c0 = rcol[r], c1 = rcol[r + 1];
                    for (int32_t i = ri[r]; i < ri[r + 1]; ++i) {
                        q.src[q.n_src++] = { w[i]->w, static_cast<int32_t>(w[i]->n_out) };
                    }
                    fn::dense::pt_args & t = q.t;
                    t.n                    = c1 - c0;
                    t.k                    = kk;
                    t.rows                 = n;
                    t.x                    = x;
                    t.row_scale            = row_scale;
                    t.err                  = a.err;
                    t.err_bit              = e16;
                    if (o16 != nullptr) {
                        t.seg[0] = { nullptr, segs[0].ld, t.n };
                        t.out16  = o16 + c0;
                        t.ld16   = l16;
                    } else {
                        t.n_seg = 0;
                        for (int32_t g = 0, g0 = 0; g < n_seg; g0 = segs[g].end, ++g) {
                            const int32_t lo = std::max(g0, c0), hi = std::min(segs[g].end, c1);
                            if (lo < hi) {
                                t.seg[t.n_seg++] = { segs[g].out + int64_t(lo - g0), segs[g].ld, hi - c0 };
                            }
                        }
                    }
                    if (!ltc(q, cfg[r])) {
                        return false;
                    }
                }
                return true;
            };
            // BF16 rows of K = 2,560 to FP16 at at, their column scales at s_at (conv: at the sublayer's piece 0)
            const auto to16b = [&](const uint16_t * bw, int32_t rows_w, int64_t at, int64_t s_at, bool conv) {
                return !conv || fn::dense::pd_convert_bf16(bw, rows_w, D, a.w16 + at,
                                                           reinterpret_cast<float *>(a.w16 + s_at), s);
            };
            const auto bkey = [&](int32_t nn) {
                return prompt_dense_key(static_cast<int32_t>(fn::dense::fmt::bf16),
                                        static_cast<int32_t>(fn::dense::pd_layout::dp2), D, nn);
            };
            // a BF16 product of rows b0 [n0] then b1 [n1] (nn columns, zero past n0 + n1): the retained route
            // converted at at with scales at s_at, or a ladder tile from the prepared scales prep
            const auto bprod = [&](const uint16_t * b0, int32_t n0, const uint16_t * b1, int32_t n1, int32_t nn,
                                   int64_t at, int64_t s_at, const float * prep, const fn::dense::pd_act & x,
                                   const fn::dense::pt_seg * segs, int32_t n_seg) {
                const fn::model::lad_cfg cfg  = lad_route(bkey(nn), n, force_.dense);
                const bool               conv = dconv(bkey(nn));
                if (!to16b(b0, n0, at, s_at, conv) ||
                    (b1 != nullptr && !to16b(b1, n1, at + int64_t(n0) * D, s_at + int64_t(2) * n0, conv))) {
                    return false;
                }
                if (cfg.retained()) {
                    return tprod(at, nn, D, x, segs, n_seg, s_at);
                }
                fn::dense::pq_args q;
                q.b0                   = b0;
                q.n0                   = n0;
                q.b1                   = b1;
                q.n1                   = n1;
                q.b_scale              = prep;
                fn::dense::pt_args & t = q.t;
                t.n                    = nn;
                t.k                    = D;
                t.rows                 = n;
                t.x                    = x;
                t.n_seg                = n_seg;
                for (int32_t i = 0; i < n_seg; ++i) {
                    t.seg[i] = segs[i];
                }
                t.col_scale = prep;
                t.err       = a.err;
                return ltc(q, cfg);
            };
            if (sub == 0 && gdn) {
                bump b{m_->single ? a_[0].scratch : sc};
                // PB3: qkv and z as the product's FP16 output, one [n][GC + GV] array (z from column GC)
                uint16_t * qz16 = b.take<uint16_t>(int64_t(n) * (GC + GV) * 2);
                float * conv = b.take<float>(int64_t(n) * GC * 4);
                uint16_t * o16 = b.take<uint16_t>(int64_t(n) * GV * 2); // PB3: o FP16
                float * al = b.take<float>(int64_t(n) * 24 * 4);
                float * be = b.take<float>(int64_t(n) * 24 * 4);
                float * g = b.take<float>(int64_t(n) * 24 * 4);
                float * beta = b.take<float>(int64_t(n) * 24 * 4);
                uint16_t * gated16 = b.take<uint16_t>(int64_t(n) * GV * 2);
                float * wy =
                    m_->single ? reinterpret_cast<float *>(a_[1].scratch) : b.take<float>(fn::gdn::prompt_wy_bytes(n));
                if (w.qkv.n_out != GC || w.z.n_out != GV || w.ssm_out.n_out != D || w.ssm_out.k != GV) {
                    return fail(err, "prompt: GDN weight shapes");
                }
                const fn::dense::pt_seg in[1] = {{nullptr, GC + GV, GC + GV}};
                const fn::dense::pt_seg ab[2] = {{al, 24, 24}, {be, 24, 48}};
                if (!dprod({ &w.qkv, &w.z }, 0, D, mx, in, 1, qz16, GC + GV, 1u) ||
                    !bprod(w.alpha, 24, w.beta, 24, 48, gdn_ab_at, gdn_ab_s_at, a.ab_s[size_t(l)], mx, ab, 2)) {
                    return fail(err, "prompt: GDN input");
                }
                count(psite::gdn_input);
                fn::gdn::prompt_args ga;
                ga.rows     = n;
                ga.qkv16 = qz16;
                ga.z16 = qz16 + GC;
                ga.ld16 = GC + GV;
                ga.conv_out = conv;
                ga.fuse_norm = true;
                ga.conv_w   = w.conv1d;
                ga.hist     = st.conv;
                ga.state    = st.gdn;
                ga.rec       = st.rec;  // ME11: piece 0 replays the selector's records
                ga.gsel     = p.c.gsel;
                ga.cur      = p.c.cur;
                ga.v_in      = prompt_piece_v_in(h);  // MUP2: the halves' 0 -> 1 -> 2, then alternating
                ga.v_out     = prompt_piece_v_out(h);
                ga.alpha    = al;
                ga.beta_raw = be;
                ga.dt       = w.dt;
                ga.a        = w.a;
                ga.g        = g;
                ga.beta     = beta;
                ga.o = nullptr;
                ga.o16 = o16;
                ga.norm_w   = w.ssm_norm;
                ga.gated16  = gated16;
                ga.err      = a.err;
                ga.wy       = wy;
                if (!fn::gdn::gdn_prompt_conv(ga, s) || !fn::gdn::gdn_prompt_prep(ga, s) ||
                    !fn::gdn::gdn_prompt_recur(ga, s) || !fn::gdn::gdn_prompt_gate(ga, s)) {
                    return fail(err, "prompt: GDN core");
                }
                note(&route_counts::gdn);
                count(psite::gdn_core);
                const fn::dense::pt_seg po[1] = {{nullptr, D, D}};
                if (!dprod({ &w.ssm_out }, gdn_out_at, GV, act1(gated16, GV), po, 1, pay)) {
                    return fail(err, "prompt: GDN out");
                }
                count(psite::gdn_out);
            } else if (sub == 0) {
                bump      b{ sc };
                float *   qfull = b.take<float>(int64_t(n) * QF * 4);
                float *   kk    = b.take<float>(int64_t(n) * 256 * 4);
                float *   vv    = b.take<float>(int64_t(n) * 256 * 4);
                float *   iq    = b.take<float>(int64_t(n) * 512 * 4);
                float *   ik    = b.take<float>(int64_t(n) * 128 * 4);
                uint8_t * wsb   = b.take<uint8_t>(static_cast<int64_t>(fn::qsa::prompt_ws_bytes(fn::qsa::max_cells)));
                (void) b.take<uint16_t>(int64_t(n) * GV * 2);  // the attention's output (phase 2)
                if (w.wq.n_out != QF || w.wk.n_out != 256 || w.wv.n_out != 256 || w.wo.n_out != D || w.wo.k != GV) {
                    return fail(err, "prompt: QSA weight shapes");
                }
                const fn::dense::pt_seg in[3] = {{qfull, QF, QF}, {kk, 256, QF + 256}, {vv, 256, QF + 512}};
                if (!dprod({ &w.wq, &w.wk, &w.wv }, 0, D, mx, in, 3) ||
                    !to16(w.wo, qsa_out_at, dconv(prompt_dense_key(w.wo.fmt, 0, GV, D)))) {
                    return fail(err, "prompt: QSA input");
                }
                const fn::dense::pt_seg ix[2] = {{iq, 512, 512}, {ik, 128, 640}};
                if (k == w.index_owner &&
                    !bprod(w.idx_q, 512, w.idx_k, 128, 640, qsa_idx_at, qsa_idx_s_at, a.idx_s[size_t(l)], mx, ix, 2)) {
                    return fail(err, "prompt: QSA index input");
                }
                count(psite::qsa_input);
                if (k == w.index_owner) {
                    if (!m_->single && ls.any_copied[k] && !ev_wait(s, ls.copied[k], err, "sel reuse")) {
                        return false;
                    }
                    const int32_t        b0 = static_cast<int32_t>(pos0 / fn::qsa::ratio);
                    const int32_t        b1 = static_cast<int32_t>((pos0 + n) / fn::qsa::ratio);
                    // ME6 (the host-backed tier, kv_tier.h): the chunk's keys in the key arena, from its first
                    // block's cells (staged once a chunk), each half's new cells written back below
                    uint16_t *      keys = st.idx_keys;
                    kv_tier * const kt   = p.c.kv;
                    if (kt != nullptr) {
                        if (h == 0 && !kt->stage_keys(k, st.kv_attn, c.start, c.rows, s, err)) {
                            return false;
                        }
                        keys = kt->stage_keys_ptr(k);
                    }
                    fn::qsa::update_args ua;
                    ua.keys      = keys;
                    ua.k_norm    = w.idx_k_norm;
                    ua.summaries = st.idx_summaries;
                    ua.first     = b0;
                    ua.count     = b1 - b0;
                    ua.rope      = fn::qsa::model_rope();
                    ua.rope_off  = c.rope_off;  // MUP1 piece 5
                    // VIS1: each block at its first row's planes, each query at its row's
                    ua.planes = c.bplanes[k] != nullptr ? c.bplanes[k] + 4 * (b0 - c.start / fn::qsa::ratio) : nullptr;
                    fn::qsa::qsp_index_args ia;
                    ia.summaries = st.idx_summaries;
                    ia.iq        = iq;
                    ia.q_norm    = w.idx_q_norm;
                    ia.rope      = fn::qsa::model_rope();
                    ia.rows      = n;
                    ia.pos0      = static_cast<int32_t>(pos0);
                    ia.rope_off  = c.rope_off;
                    ia.planes    = c.planes[k] != nullptr ? c.planes[k] + 4 * (pos0 - c.start) : nullptr;
                    ia.max_cells = st.n_cells;
                    ia.ws        = fn::qsa::prompt_carve(wsb, st.n_cells);
                    ia.ws.error  = a.err + 1;
                    ia.body      = reinterpret_cast<uint32_t *>(a.sel_send[sh] + fn::xchg::pmsg_payload_at);
                    ia.offsets   = reinterpret_cast<int32_t *>(ia.body + fn::qsa::prompt_body_words);
                    if (!fn::qsa::qsp_keys(n, static_cast<int32_t>(pos0), ik, keys, st.n_cells, s) ||
                        (ua.count > 0 && !fn::qsa::summary_update(ua, s)) ||
                        (kt != nullptr && !kt->write_back_keys(k, st.kv_attn, pos0, n, s, err)) ||
                        !fn::qsa::qsp_select(ia, s) ||
                        !fn::xchg::pstamp(a.sel_send[sh], gen_sel, n, fn::xchg::pkind::sel, s) ||
                        (!m_->single && !ev_record(ls.prod[k], k, s, err, "sel produced"))) {
                        return err.empty() ? fail(err, "prompt: QSA index") : false;
                    }
                    ls.g_prod[k] = gen_sel;
                    count(psite::qsa_index);
                }
            } else {
                // the FFN: the router and shared gate (both cards, the full row), EP1, the shared expert
                bump b0{m_->single ? a_[0].scratch : sc};
                float * logits = b0.take<float>(int64_t(n) * NE * 4);
                float * sgl = b0.take<float>(int64_t(n) * 8 * 4); // [n][8]: column 512 of the router's product
                float * sg = b0.take<float>(int64_t(n) * 4);
                (void)b0.take<float>(int64_t(n) * 4);
                int32_t * ids = b0.take<int32_t>(int64_t(n) * NU * 4);
                float * wts = b0.take<float>(int64_t(n) * NU * 4);
                fn::experts::ep_route rt;
                rt.offsets = b0.take<int32_t>(int64_t(513) * 4);
                rt.sorted = b0.take<int32_t>(int64_t(n) * NU * 4);
                rt.pos = b0.take<int32_t>(int64_t(n) * NU * 4);
                rt.tasks = b0.take<int32_t>(int64_t(fn::experts::ep_max_tasks) * 16);
                rt.counters = b0.take<int32_t>(int64_t(8) * 4);
                int32_t * toff = b0.take<int32_t>(int64_t(513) * 4); // ME4: a ring layer's task offsets
                fn::experts::ep_lists lists;  // APF2: the grouped task lists (gate/up's tile, down's)
                int32_t * const       ltasks[2] = { b0.take<int32_t>(int64_t(fn::experts::ep_max_ltasks) * 16),
                                                    b0.take<int32_t>(int64_t(fn::experts::ep_max_ltasks) * 16) };
                bump b{sc, !m_->single || k == 0 ? b0.off : 0};
                uint16_t * glu = b.take<uint16_t>(int64_t(n) * NU * 320 * 2);
                uint16_t * part16 = b.take<uint16_t>(int64_t(n) * NU * D * 2); // PB3: the partials FP16
                float * shared = b.take<float>(int64_t(n) * D * 4);
                bump bs{m_->single ? a_[0].scratch : sc, m_->single && k == 1 ? b0.off + b.off : b.off};
                float * sh_g = bs.take<float>(int64_t(n) * 320 * 4);
                float * sh_u = bs.take<float>(int64_t(n) * 320 * 4);
                uint16_t * sh_glu = bs.take<uint16_t>(int64_t(n) * 320 * 2);

                // PB3: the shared gate's BF16 row converted after the router's, then 7 zero rows (scales 0), so one
                // tensor product gives the logits and the gate logit (column 512) from the FP16 mixed row
                if (!m_->single || k == 0) {
                    float * const rsc = reinterpret_cast<float *>(a.w16 + ffn_router_s_at);
                    const fn::dense::pt_seg rs[2] = {{logits, NE, NE}, {sgl, 8, NE + 8}};
                    if ((dconv(prompt_dense_key(static_cast<int32_t>(fn::dense::fmt::bf16), 0, D, NE + 8)) &&
                         (!cuda_ok(
                              cudaMemsetAsync(a.w16 + ffn_router_at + int64_t(NE + 1) * D, 0, size_t(7) * D * 2, s),
                              "gate pad", err) ||
                          !cuda_ok(cudaMemsetAsync(rsc + NE + 1, 0, size_t(7) * 4, s), "gate pad", err))) ||
                        !bprod(w.router, NE, w.sh_gate_w, 1, NE + 8, ffn_router_at, ffn_router_s_at, a.rt_s[size_t(l)],
                               mx, rs, 2) ||
                        !fn::dense::pd_route(n, logits, sgl, ids, wts, sg, s, 8) ||
                        (skew_ && !cuda_ok(cudaMemcpyAsync(ids, skew_host_ + int64_t(r0) * NU, size_t(n) * NU * 4,
                                                           cudaMemcpyHostToDevice, s),
                                           "skewed routes", err))) {
                        return err.empty() ? fail(err, "prompt: router") : false;
                    }
                    count(psite::ffn_router);
                }
                const auto gf = static_cast<fn::experts::fmt>(w.ex_gu_fmt);
                const auto df = static_cast<fn::experts::fmt>(w.ex_down_fmt);
                fn::experts::ep_gate_up_args gu;
                gu.w       = w.ex_gate_up;
                gu.grids   = p.c.grids;
                gu.x       = a.mixed16;
                gu.route   = rt;
                gu.glu     = glu;
                gu.glu_f32 = nullptr;
                fn::experts::ep_down_args dn;
                dn.w     = w.ex_down;
                dn.grids = p.c.grids;
                dn.glu   = glu;
                dn.route = rt;
                dn.part = nullptr;
                dn.part16 = part16;
                dn.err = a.err;
                dn.err_bit = fn::xchg::err_prange;
                // ME1: the expert cache's staged layer (a one-half chunk: its routed experts, from card 0's routing)
                expert_service * const xs = w.ex_gate_up == nullptr ? p.c.ex_service : nullptr;
                bool ring = false; // ME4: the layer's experts come through the service's ring, group by group
                if (xs != nullptr) {
                    void * const streams[2] = { p_[0]->c.stream, p_[1]->c.stream };
                    const int32_t * const rid        = prompt_pieces(c) == 1 ? ids : nullptr;
                    if (h == 0 && k == 0 && !xs->prompt_layer(l, n, rid, c.more, streams, err)) {
                        return err.empty() ? fail(err, "prompt: expert staging") : false;
                    }
                    ring = xs->prompt_groups(l) > 0;
                    // MX7: the CPU share's experts posted (their rows gathered from card 0's input; one device)
                    if (ring && k == 0) {
                        settle.xs = xs; // from here every exit of the layer joins the share's job
                        if (!xs->prompt_share_begin(l, a.mixed16, streams, err)) {
                            return err.empty() ? fail(err, "prompt: the CPU share") : false;
                        }
                    }
                    gu.w = xs->prompt_bank(k);
                    dn.w = xs->prompt_bank(k);
                    if (!ring) {
                        gu.wt = xs->prompt_table(l, k, false);
                        dn.wt = xs->prompt_table(l, k, true);
                        if (gu.wt == nullptr || dn.wt == nullptr) {
                            return fail(err, "prompt: the expert cache has no staged table of the layer");
                        }
                    }
                }
                // MX7: on a ring layer the CPU share's experts' tasks dropped, so the groups launch only the GPU's.
                // MX6 (single card): card 1 reads card 0's routing tables, so the grouping, the drop and the
                // offsets run once, on card 0
                const uint8_t * const skip = ring ? xs->prompt_share_skip(l) : nullptr;
                const bool own = !m_->single || k == 0;
                // APF2: gate/up and down each grouped on the ladder's route at the piece's routed rows an expert
                // (resident Q4_K / Q5_K gate/up, Q5_1 / Q8_0 down; the expert service's layers keep per-task), over
                // ep_group's list for the route's tile
                const int32_t            er   = prompt_expert_rows(n);
                const fn::model::lad_cfg gcf =
                    xs == nullptr && fn::experts::el_takes_gate_up(gf) ?
                        lad_route(prompt_expert_key(false, static_cast<int32_t>(gf)), er, force_.expert) :
                        fn::model::lad_cfg{};
                const fn::model::lad_cfg dcf =
                    xs == nullptr && fn::experts::el_takes_down(df) ?
                        lad_route(prompt_expert_key(true, static_cast<int32_t>(df)), er, force_.expert) :
                        fn::model::lad_cfg{};
                if (!gcf.retained()) {
                    lists.tasks[0] = ltasks[0];
                    lists.rows[0]  = fn::experts::el_rows(gcf.tile);
                }
                if (!dcf.retained()) {
                    lists.tasks[1] = ltasks[1];
                    lists.rows[1]  = fn::experts::el_rows(dcf.tile);
                }
                if (!ring) {
                    note(!gcf.retained() ? &route_counts::ex_grouped : &route_counts::ex_per_task);
                    note(!dcf.retained() ? &route_counts::exd_grouped : &route_counts::exd_per_task);
                    note_fmt(&route_counts::ex_fmts, static_cast<int32_t>(gf));
                }
                if ((own && !fn::experts::ep_group(n, fn::experts::ep_task_rows(gf), ids, rt, a.err + 2, s, lists)) ||
                    (own && skip != nullptr && !fn::experts::es_drop(rt, skip, s)) ||
                    (ring ? (own && !fn::experts::ep_task_offsets(rt, toff, s)) :
                            !(gcf.retained() ? fn::experts::ep_gate_up(gf, n, gu, s) :
                                               fn::experts::ep_gate_up_ladder(gf, gcf.tile, n, gu, ltasks[0],
                                                                              rt.counters + 2, s)))) {
                    return fail(err, "prompt: experts gate/up");
                }
                // the shared expert: gate and up (DP2's packing), the GLU, its down (EX1's) scaled by sg
                if (w.sh_gate.n_out != 320 || w.sh_up.n_out != 320) {
                    return fail(err, "prompt: shared expert shapes");
                }
                const fn::dense::pt_seg gu2[2] = {{sh_g, 320, 320}, {sh_u, 320, 640}};
                if (!dprod({ &w.sh_gate, &w.sh_up }, 0, D, mx, gu2, 2) ||
                    !fn::dense::pd_glu(n, 320, sh_g, sh_u, 320, sh_glu, 320, s)) {
                    return fail(err, "prompt: shared expert");
                }
                {
                    // the shared down (EX1's layout), scaled by sg after the sum
                    dense_weight sd;
                    sd.w                          = w.ex_sh_down;
                    sd.fmt                        = static_cast<int32_t>(dense_of(w.ex_sh_down_fmt));
                    sd.k                          = 320;
                    sd.n_out                      = D;
                    const fn::dense::pt_seg so[1] = {
                        { shared, D, D }
                    };
                    if (!dprod({ &sd }, ffn_down_at, 320, act1(sh_glu, 320), so, 1, nullptr, D, fn::xchg::err_prange,
                               fn::dense::pd_layout::ex1_down, sg)) {
                        return fail(err, "prompt: shared down");
                    }
                }
                count(psite::ffn_shared);
                fn::experts::ep_sum_args sa;
                sa.part = nullptr;
                sa.part16 = part16;
                sa.route  = rt;
                sa.wts    = wts;
                sa.shared = shared;
                sa.out = nullptr;
                sa.out16 = pay;
                sa.err = a.err;
                sa.err_bit = fn::xchg::err_prange;
                if (ring) {
                    rcard[k] = { true, gf, df, gu, dn, sa, toff };
                } else {
                    if (!(dcf.retained() ?
                              fn::experts::ep_down(df, n, dn, s) :
                              fn::experts::ep_down_ladder(df, dcf.tile, n, dn, ltasks[1], rt.counters + 3, s)) ||
                        !fn::experts::ep_sum(n, sa, s)) {
                        return fail(err, "prompt: experts down");
                    }
                    count(psite::ffn_experts);
                }
            }
        } else {
            // phase 2: the body's rest, then the partial into the P message
            if (end) {
                continue;
            }
            const layer_weights & w  = p.w[l];
            const layer_state &   st = p.s[l];
            if (sub == 0 && !gdn) {
                bump    b{ a.scratch };
                float * qfull = b.take<float>(int64_t(n) * QF * 4);
                float * kk    = b.take<float>(int64_t(n) * 256 * 4);
                float * vv    = b.take<float>(int64_t(n) * 256 * 4);
                (void) b.take<float>(int64_t(n) * 512 * 4);
                (void) b.take<float>(int64_t(n) * 128 * 4);
                (void) b.take<uint8_t>(static_cast<int64_t>(fn::qsa::prompt_ws_bytes(fn::qsa::max_cells)));
                uint16_t *      att16 = b.take<uint16_t>(int64_t(n) * GV * 2);
                const bool      owner = k == w.index_owner;
                const uint8_t * sel   = owner ? a.sel_send[sh] : a.sel_recv[sh];
                // ME1 (the host-backed KV, kv_tier.h): the layer's history below the chunk staged into the card's
                // arenas before the first half, which then serve as its whole cache; each half's new cells go back
                uint16_t *      kc = st.k_cache;
                uint16_t *      vc = st.v_cache;
                kv_tier * const kt = p.c.kv;
                if (kt != nullptr) {
                    if (h == 0 && !kt->stage(k, st.kv_attn, pos0, s, err)) {
                        return false;
                    }
                    kc = kt->stage_k(k);
                    vc = kt->stage_v(k);
                }
                // PB3: the K/V cache writes need no selection: on the peer they run ahead of its wait for SEL
                fn::qsa::qsp_kv_args ka;
                ka.kv_k = kk;
                ka.kv_v = vv;
                ka.k_norm = w.k_norm;
                ka.rows = n;
                ka.pos0 = static_cast<int32_t>(pos0);
                ka.rope_off  = c.rope_off;  // MUP1 piece 5
                ka.planes    = c.planes[k] != nullptr ? c.planes[k] + 4 * (pos0 - c.start) : nullptr;
                ka.max_cells = st.n_cells;
                ka.k_cache = kc;
                ka.v_cache = vc;
                ka.k_scale = st.k_scale; // KV8: set in q8 (never with the host-backed tier)
                ka.v_scale = st.v_scale;
                if (!fn::qsa::qsp_kv(ka, s)) {
                    return fail(err, "prompt: QSA K/V");
                }
                if (kt != nullptr && !kt->write_back(k, st.kv_attn, pos0, n, s, err)) {
                    return false;
                }
                if (!owner) {
                    if (ls.g_copied[1 - k] != gen_sel) {
                        return fail(err, "prompt: a selection received before its generation was copied");
                    }
                    if ((!m_->single && !ev_wait(s, ls.copied[1 - k], err, "sel recv")) ||
                        !fn::xchg::pcheck(sel, gen_sel, a.err, s)) {
                        return err.empty() ? fail(err, "prompt: sel check") : false;
                    }
                }
                const uint32_t *     body = reinterpret_cast<const uint32_t *>(sel + fn::xchg::pmsg_payload_at);
                fn::qsa::qsp_attend_args aa;
                aa.qfull     = qfull;
                aa.q_norm    = w.q_norm;
                aa.body      = body;
                aa.offsets   = reinterpret_cast<const int32_t *>(body + fn::qsa::prompt_body_words);
                aa.k_cache   = kc;
                aa.v_cache   = vc;
                aa.k_scale = st.k_scale;
                aa.v_scale = st.v_scale;
                aa.rows      = n;
                aa.pos0      = static_cast<int32_t>(pos0);
                aa.rope_off  = ka.rope_off;
                aa.planes    = ka.planes;
                aa.max_cells = st.n_cells;
                aa.out16     = att16;
                aa.err       = a.err;
                if (!fn::qsa::qsp_attend(aa, s)) {
                    return fail(err, "prompt: QSA attention");
                }
                note(&route_counts::qsa);
                if (!owner) {
                    if (!m_->single && !ev_record(ls.freed[k], k, s, err, "sel freed")) {
                        return false;
                    }
                    ls.g_freed[k] = gen_sel;
                    ls.any_freed[k] = true;
                }
                count(psite::qsa_attend);
                // wo: the ladder's route at the piece's rows, else the conversion phase 0 made (APF2)
                fn::dense::pt_args t;
                t.w      = a.w16 + qsa_out_at;
                t.n      = D;
                t.k      = GV;
                t.rows   = n;
                t.x      = act1(att16, GV);
                t.seg[0] = {nullptr, D, D};
                t.out16   = reinterpret_cast<uint16_t *>(p_msg(k, true, c, h) + fn::xchg::pmsg_payload_at);
                t.ld16 = D;
                t.err = a.err;
                t.err_bit = fn::xchg::err_prange;
                const fn::model::lad_cfg wcfg = lad_route(prompt_dense_key(w.wo.fmt, 0, GV, D), n, force_.dense);
                bool                     wo_ok = false;
                if (!wcfg.retained()) {
                    fn::dense::pq_args q;
                    q.f      = static_cast<fn::dense::fmt>(w.wo.fmt);
                    q.src[0] = { w.wo.w, D };
                    q.n_src  = 1;
                    q.grids  = p.c.dense_grids;
                    q.t      = t;
                    q.t.w    = nullptr;
                    note(&route_counts::dense_ladder);
                    if (wcfg.splits > 1) {
                        note(&route_counts::dense_split);
                    }
                    wo_ok = fn::dense::pl_tc(q, wcfg.tile, wcfg.splits, a.lws, s);
                } else {
                    note(&route_counts::dense_retained);
                    wo_ok = fn::dense::pd_tc(t, s);
                }
                if (!wo_ok) {
                    return fail(err, "prompt: QSA out");
                }
                count(psite::qsa_out);
            }
            // the payload written by the sublayer's producer (phase 0 waited for the slot's previous copy)
            if (!fn::xchg::pseal(n, p_msg(k, true, c, h), gen, s) ||
                (!m_->single && !ev_record(lp.prod[k], k, s, err, "produced"))) {
                return err.empty() ? fail(err, "prompt: pack") : false;
            }
            lp.g_prod[k] = gen;
        }
    }
    if (rcard[0].on || rcard[1].on) {
        // ME4: the ring layer's groups in order. Each: its weights and tables acquired on every device (their copies
        // after every earlier reader of those ring bytes), both logical cards' gate/up and down over the group's tasks,
        // then its release after those readers. The groups partition the experts and so ep_group's tasks: each task
        // runs once, and each card's ep_sum after the last group adds all ten routed terms in route order
        expert_service * const xs        = p_[0]->c.ex_service;
        void * const           streams[2] = { p_[0]->c.stream, p_[1]->c.stream };
        const int32_t          ng         = xs != nullptr ? xs->prompt_groups(l) : 0;
        if (!rcard[0].on || !rcard[1].on || ng < 1) {
            return fail(err, "prompt: a ring layer without both logical cards or its groups");
        }
        for (int32_t g = 0; g < ng; ++g) {
            prompt_group_view v;
            if (!xs->prompt_group(l, g, streams, v, err)) {
                return err.empty() ? fail(err, "prompt: ring group") : false;
            }
            for (int32_t k = 0; k < 2; ++k) {
                const ring_card & r  = rcard[k];
                const cudaStream_t s = static_cast<cudaStream_t>(p_[k]->c.stream);
                const int64_t      tr = fn::experts::ep_task_rows(r.gf);
                const int64_t      tasks = int64_t(v.e_hi - v.e_lo) + (v.pairs + tr - 1) / tr;
                fn::experts::ep_gate_up_args gu = r.gu;
                fn::experts::ep_down_args    dn = r.dn;
                gu.wt       = v.table[k][0];
                dn.wt       = v.table[k][1];
                gu.task_off = dn.task_off = r.toff;
                gu.e_lo = dn.e_lo = v.e_lo;
                gu.e_hi = dn.e_hi = v.e_hi;
                gu.grid = dn.grid = static_cast<int32_t>(std::min<int64_t>(fn::experts::ep_max_tasks, tasks));
                if (!cuda_ok(cudaSetDevice(m_->device[k]), "device", err) || !fn::experts::ep_gate_up(r.gf, n, gu, s) ||
                    !fn::experts::ep_down(r.df, n, dn, s)) {
                    return err.empty() ? fail(err, "prompt: ring group experts") : false;
                }
            }
            if (!xs->prompt_group_done(l, g, streams, err)) {
                return err.empty() ? fail(err, "prompt: ring group release") : false;
            }
        }
        // MX7: the CPU share's products into each card's partials (card 0 the whole products, card 1 zeros)
        prompt_share_card sc[2];
        for (int32_t k = 0; k < 2; ++k) {
            sc[k] = {rcard[k].sa.route.pos, rcard[k].dn.part16, rcard[k].dn.err, rcard[k].dn.err_bit, rcard[k].sa.wts};
        }
        if (!xs->prompt_share_end(l, sc, streams, err)) {
            return err.empty() ? fail(err, "prompt: the CPU share's add-back") : false;
        }
        for (int32_t k = 0; k < 2; ++k) {
            if (!cuda_ok(cudaSetDevice(m_->device[k]), "device", err) ||
                !fn::experts::ep_sum(n, rcard[k].sa, static_cast<cudaStream_t>(p_[k]->c.stream))) {
                return err.empty() ? fail(err, "prompt: experts sum") : false;
            }
            count(psite::ffn_experts);
        }
    }
    if (phase == 0 && l == 1 && sub == 0) {
        hist_cur_ ^= 1;  // the next piece's PLE reads this piece's history
    }
    return true;
}

// The draft's catch-up over half h's positions, both cards:
//   0: e and h normalized (FP16), eh_proj, the split HC's down: this card's moments into the MOM message
//   1: the MOM copy
//   2: the peer's moments acquired, HC mid and up: this card's mixed half into the DMIX message
//   3: the DMIX copy
//   4: the peer's half acquired, the mixed row, K and V, the draft cache
bool prompt_path::draft_half(const prompt_chunk & c, int32_t h, int32_t phase, std::string & err) {
    const int32_t  r0   = h == 0 ? 0 : c.half[0];
    const int32_t  n    = c.half[h];
    const int64_t  pos0 = c.start + r0;
    link &         lm   = links_[lk_mom];
    link &         ld   = links_[lk_dmix];
    const uint64_t gmom = fn::xchg::pgen(c.seq, fn::n_layer, 0, h, fn::xchg::pkind::mom);
    const uint64_t gmix = fn::xchg::pgen(c.seq, fn::n_layer, 0, h, fn::xchg::pkind::dmix);
    if (phase == 1 || phase == 3) {
        link &         lk    = phase == 1 ? lm : ld;
        const uint64_t g = phase == 1 ? gmom : gmix;
        const int64_t bytes = fn::xchg::pmsg_bytes(phase == 1 ? int64_t(n) * fn::hc::hcp_mom * 4 : int64_t(n) * F * 4);
        for (int32_t k = 0; k < 2; ++k) {
            if (lk.g_prod[k] != g) {
                return fail(err, "prompt: a draft message copied before it was produced");
            }
            if (!m_->single) {
                cudaStream_t cs = static_cast<cudaStream_t>(copy_[k]);
                if (!cuda_ok(cudaSetDevice(m_->device[k]), "device", err) ||
                    !cuda_ok(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(lk.prod[k]), 0), "copy wait", err) ||
                    (lk.any_freed[1 - k] &&
                     !cuda_ok(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(lk.freed[1 - k]), 0), "copy free",
                              err)) ||
                    !cuda_ok(cudaMemcpyPeerAsync(lk.recv[1 - k], m_->device[1 - k], lk.send[k], m_->device[k],
                                                 size_t(bytes), cs),
                             "copy", err) ||
                    !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(lk.copied[k]), cs), "copied", err)) {
                    return false;
                }
            }
            lk.g_copied[k] = g;
            lk.any_copied[k] = true;
        }
        return true;
    }
    for (int32_t k = 0; k < 2; ++k) {
        card_arena & a = a_[size_t(k)];
        pass_card &  p = *p_[k];
        draft_card & d = *d_[k];
        cudaStream_t s = static_cast<cudaStream_t>(p.c.stream);
        if (!cuda_ok(cudaSetDevice(m_->device[k]), "device", err)) {
            return false;
        }
        bump             b{ a.scratch };
        const auto       to16d = [&](const dense_weight & dw, int64_t at, bool conv) {
            if (!conv) {
                return true;
            }
            fn::dense::pd_args pa;
            pa.f = static_cast<fn::dense::fmt>(dw.fmt);
            pa.lay = fn::dense::pd_layout::dp2;
            pa.k = static_cast<int32_t>(dw.k);
            pa.n = static_cast<int32_t>(dw.n_out);
            pa.w = dw.w;
            pa.grids = p.c.dense_grids;
            return fn::dense::pd_convert(pa, a.w16 + at, s);
        };
        // APF2: pt's key (dw's format, or the draft index's BF16 rows with dw null), and its route at pt's rows: pd_tc
        // over the conversion (pt.w), or pl_tc from dw itself (a packed matrix) or the BF16 rows and their prepared
        // scales
        const auto dkey = [&](const dense_weight * dw, int32_t kk, int32_t nn) {
            return prompt_dense_key(dw != nullptr ? dw->fmt : static_cast<int32_t>(fn::dense::fmt::bf16), 0, kk, nn);
        };
        const auto dtc = [&](const fn::dense::pt_args & pt, const dense_weight * dw) {
            const fn::model::lad_cfg cfg = lad_route(dkey(dw, pt.k, pt.n), pt.rows, force_.dense);
            if (cfg.retained()) {
                note(&route_counts::dense_retained);
                return fn::dense::pd_tc(pt, s);
            }
            fn::dense::pq_args q;
            if (dw != nullptr) {
                q.f      = static_cast<fn::dense::fmt>(dw->fmt);
                q.src[0] = { dw->w, static_cast<int32_t>(dw->n_out) };
                q.n_src  = 1;
                q.grids  = p.c.dense_grids;
            } else {
                q.b0      = d.iw.k_proj;
                q.n0      = 128;
                q.b_scale = a.dik_s;
            }
            q.t   = pt;
            q.t.w = nullptr;
            if (dw == nullptr) {
                q.t.col_scale = a.dik_s;
            }
            note(&route_counts::dense_ladder);
            if (cfg.splits > 1) {
                note(&route_counts::dense_split);
            }
            return fn::dense::pl_tc(q, cfg.tile, cfg.splits, a.lws, s);
        };
        // whether piece 0 converts the product of key (some piece of the chunk takes the retained route; mul: the
        // product's rows a piece row)
        const auto dcv = [&](const fn::model::lad_key & key, int32_t mul) {
            return h == 0 && lad_retained_any(key, c, force_.dense, mul);
        };
        uint16_t *       e16     = b.take<uint16_t>(int64_t(n) * D * 2);
        uint16_t *       h16     = b.take<uint16_t>(int64_t(n) * 4 * D * 2);
        float *          eh      = b.take<float>(int64_t(n) * 4 * F * 4);
        float *          low     = b.take<float>(int64_t(n) * 320 * 4);
        float *          rinv    = b.take<float>(int64_t(n) * 4 * 4);
        float *          inj     = b.take<float>(int64_t(n) * 4 * 4);
        uint16_t *       mixed16 = b.take<uint16_t>(int64_t(n) * D * 2);
        float *          kk      = b.take<float>(int64_t(n) * 256 * 4);
        float *          vv      = b.take<float>(int64_t(n) * 256 * 4);
        float *          ik      = b.take<float>(int64_t(n) * 128 * 4); // SP2 (sparse): the index key
        fn::hc::hcp_args ha;
        ha.w.form      = fn::hc::hcp_form::split;
        ha.w.q8_down   = d.w.hc_attn.q8_down;
        ha.w.q8_up     = d.w.hc_attn.q8_up;
        ha.w.gamma_own = d.w.hc_attn.gamma;
        ha.rows        = n;
        ha.x           = eh;
        ha.mom         = reinterpret_cast<float *>(a.mom_send + fn::xchg::pmsg_payload_at);
        ha.mom_peer    = reinterpret_cast<const float *>(a.mom_recv + fn::xchg::pmsg_payload_at);
        ha.card        = k;
        ha.low         = low;
        ha.rinv        = rinv;
        ha.inject      = inj;
        ha.mixed       = reinterpret_cast<float *>(a.dmix_send + fn::xchg::pmsg_payload_at);
        ha.ld_mixed    = F;
        ha.err         = a.err;
        if (phase == 0) {
            // VIS1: one device's two logical cards share E (set_links)
            if (c.n_runs > 0 && (k == 0 || a.E != a_[0].E) && !draft_image_rows(c, k, r0, n, a.E, s, err)) {
                return false;
            }
            fn::draft::norm_args na;
            na.rows   = n;
            na.emb    = a.E + int64_t(r0) * D;
            na.hidden = a.S + int64_t(r0) * 4 * D;
            na.pred   = r0 == 0 ? a.pred : a.S + int64_t(r0 - 1) * 4 * D;
            na.enorm  = a.enorm;
            na.hnorm  = a.hnorm;
            na.e16    = e16;
            na.h16    = h16;
            na.err    = a.err;
            // PB3: eh_proj, wk and wv to FP16 in the conversion region at the first half, then eh_proj on tensor cores
            // over the [e | h_s] rows (four stream rows a position, e shared)
            if (d.w.eh.n_out != F || d.w.eh.k != 2 * D || d.w.wk.n_out != 256 || d.w.wv.n_out != 256 || d.w.wk.k != D ||
                d.w.wv.k != D) {
                return fail(err, "prompt: draft weight shapes");
            }
            fn::dense::pt_args pt;
            pt.w = a.w16;
            pt.n = F;
            pt.k = 2 * D;
            pt.rows = 4 * n;
            pt.x.a0 = e16;
            pt.x.ld0 = D;
            pt.x.shift0 = 2;
            pt.x.a1 = h16;
            pt.x.ld1 = D;
            pt.x.shift1 = 0;
            pt.x.k_split = D;
            pt.seg[0] = {eh, F, F};
            pt.n_seg = 1;
            // APF2: each conversion where some piece takes the retained route
            bool cv      = to16d(d.w.eh, 0, dcv(dkey(&d.w.eh, 2 * D, F), 4)) &&
                           to16d(d.w.wk, draft_wk_at, dcv(dkey(&d.w.wk, D, 256), 1)) &&
                           to16d(d.w.wv, draft_wv_at, dcv(dkey(&d.w.wv, D, 256), 1));
            if (d.sparse && dcv(dkey(nullptr, D, 128), 1)) {
                // SP2 (sparse): the index k's FP16 copy and its column scales too
                cv = cv && fn::dense::pd_convert_bf16(d.iw.k_proj, 128, D, a.w16 + draft_ik_at,
                                                      reinterpret_cast<float *>(a.w16 + draft_ik_s_at), s);
            }
            if (!m_->single && lm.any_copied[k] &&
                !cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(lm.copied[k]), 0), "mom reuse", err)) {
                return false;
            }
            if (test_h_ != nullptr && (!m_->single || k == 0) &&
                (pos0 + n > test_h_rows_ || !cuda_ok(cudaMemcpyAsync(a.S + int64_t(r0) * 4 * D, test_h_ + pos0 * 4 * D,
                                                                     size_t(n) * 4 * D * 4, cudaMemcpyHostToDevice, s),
                                                     "test hidden rows", err))) {
                return err.empty() ? fail(err, "prompt: test hidden rows beyond the given rows") : false;
            }
            if (!cv || !fn::draft::dcp_norm(na, s) || !dtc(pt, &d.w.eh) || !fn::hc::hcp_down(ha, s) ||
                !fn::xchg::pstamp(a.mom_send, gmom, n, fn::xchg::pkind::mom, s) ||
                (!m_->single &&
                 !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(lm.prod[k]), s), "mom produced", err))) {
                return err.empty() ? fail(err, "prompt: draft input") : false;
            }
            lm.g_prod[k] = gmom;
            if (test_eh_[k] != nullptr &&
                (pos0 + n > test_m_rows_ ||
                 !cuda_ok(cudaMemcpyAsync(test_eh_[k] + pos0 * 4 * F, eh, size_t(n) * 4 * F * 4,
                                          cudaMemcpyDeviceToHost, s),
                          "test eh rows", err))) {
                return err.empty() ? fail(err, "prompt: test eh rows beyond the given rows") : false;
            }
        } else if (phase == 2) {
            if (lm.g_copied[1 - k] != gmom) {
                return fail(err, "prompt: draft moments received before they were copied");
            }
            if (!m_->single && ld.any_copied[k] &&
                !cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(ld.copied[k]), 0), "dmix reuse", err)) {
                return false;
            }
            if ((!m_->single &&
                 !cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(lm.copied[1 - k]), 0), "mom recv", err)) ||
                !fn::xchg::pcheck(a.mom_recv, gmom, a.err, s) || !fn::hc::hcp_mid(ha, s) ||
                (!m_->single &&
                 !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(lm.freed[k]), s), "mom freed", err)) ||
                !fn::hc::hcp_up(ha, s) || !fn::xchg::pstamp(a.dmix_send, gmix, n, fn::xchg::pkind::dmix, s) ||
                (!m_->single &&
                 !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ld.prod[k]), s), "dmix produced", err))) {
                return err.empty() ? fail(err, "prompt: draft HC") : false;
            }
            lm.g_freed[k] = gmom;
            lm.any_freed[k] = true;
            ld.g_prod[k]    = gmix;
            if (test_low_[k] != nullptr &&
                (pos0 + n > test_m_rows_ ||
                 !cuda_ok(cudaMemcpyAsync(test_low_[k] + pos0 * 320, low, size_t(n) * 320 * 4, cudaMemcpyDeviceToHost,
                                          s),
                          "test low rows", err))) {
                return err.empty() ? fail(err, "prompt: test low rows beyond the given rows") : false;
            }
        } else {
            if (ld.g_copied[1 - k] != gmix) {
                return fail(err, "prompt: the draft's mixed half received before it was copied");
            }
            const auto q8 = [&](int64_t at, float * out, const dense_weight & dw) {  // wk or wv (phase 0's conversion)
                fn::dense::pt_args pt;
                pt.w = a.w16 + at;
                pt.n = 256;
                pt.k = D;
                pt.rows = n;
                pt.x = act1(mixed16, D);
                pt.seg[0] = {out, 256, 256};
                pt.n_seg = 1;
                return dtc(pt, &dw);
            };
            fn::draft::kvp_args kp;
            kp.rows        = n;
            kp.pos0        = static_cast<int32_t>(pos0);
            kp.max_cells   = d.max_cells;
            kp.kv_k        = kk;
            kp.kv_v        = vv;
            kp.k_norm      = d.w.k_norm;
            kp.k_cache = d.k_cache;
            kp.v_cache = d.v_cache;
            kp.theta_scale = fn::qsa::attn::rope_theta_scale();
            kp.rope_off    = c.rope_off;  // MUP1 piece 5
            kp.planes      = c.planes[k] != nullptr ? c.planes[k] + 4 * (pos0 - c.start) : nullptr;
            if ((!m_->single &&
                 !cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(ld.copied[1 - k]), 0), "dmix recv", err)) ||
                !fn::xchg::pcheck(a.dmix_recv, gmix, a.err, s) ||
                !fn::draft::dcp_mixed(n, k, reinterpret_cast<const float *>(a.dmix_send + fn::xchg::pmsg_payload_at),
                                      reinterpret_cast<const float *>(a.dmix_recv + fn::xchg::pmsg_payload_at), mixed16,
                                      a.err, s) ||
                (!m_->single &&
                 !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ld.freed[k]), s), "dmix freed", err)) ||
                !q8(draft_wk_at, kk, d.w.wk) || !q8(draft_wv_at, vv, d.w.wv) || !fn::draft::dcp_prepare(kp, s)) {
                return err.empty() ? fail(err, "prompt: draft K/V") : false;
            }
            if (test_m_[k] != nullptr &&
                (pos0 + n > test_m_rows_ ||
                 !cuda_ok(cudaMemcpyAsync(test_m_[k] + pos0 * D, mixed16, size_t(n) * D * 2, cudaMemcpyDeviceToHost, s),
                          "test mixed rows", err))) {
                return err.empty() ? fail(err, "prompt: test mixed rows beyond the given rows") : false;
            }
            if (d.sparse) {
                // SP2 (sparse draft attention): the chunk's index keys (the BF16 product on tensor cores, as the
                // target's prompt index) into the draft index, and the blocks the chunk completes pooled
                fn::dense::pt_args pt;
                pt.w = a.w16 + draft_ik_at;
                pt.n = 128;
                pt.k = D;
                pt.rows = n;
                pt.x = act1(mixed16, D);
                pt.seg[0] = {ik, 128, 128};
                pt.n_seg = 1;
                pt.col_scale = reinterpret_cast<const float *>(a.w16 + draft_ik_s_at);
                const int32_t        b0 = static_cast<int32_t>(pos0 / fn::qsa::ratio);
                const int32_t        b1 = static_cast<int32_t>((pos0 + n) / fn::qsa::ratio);
                // ME6: the draft's keys in the key arena, as the target's
                uint16_t *      keys = d.idx_keys;
                kv_tier * const kt   = p.c.kv;
                if (kt != nullptr) {
                    if (h == 0 && !kt->stage_keys(k, kv_tier::draft_attn, c.start, c.rows, s, err)) {
                        return false;
                    }
                    keys = kt->stage_keys_ptr(k);
                }
                fn::qsa::update_args ua;
                ua.keys      = keys;
                ua.k_norm    = d.iw.k_norm;
                ua.summaries = d.idx_summ;
                ua.first     = b0;
                ua.count     = b1 - b0;
                ua.rope      = fn::qsa::model_rope();
                ua.rope_off  = c.rope_off;  // MUP1 piece 5
                ua.planes    = c.bplanes[k] != nullptr ? c.bplanes[k] + 4 * (b0 - c.start / fn::qsa::ratio) : nullptr;
                if (!dtc(pt, nullptr) || !fn::qsa::qsp_keys(n, static_cast<int32_t>(pos0), ik, keys, d.max_cells, s) ||
                    (ua.count > 0 && !fn::qsa::summary_update(ua, s)) ||
                    (kt != nullptr && !kt->write_back_keys(k, kv_tier::draft_attn, pos0, n, s, err))) {
                    return err.empty() ? fail(err, "prompt: draft index") : false;
                }
            }
            ld.g_freed[k]   = gmix;
            ld.any_freed[k] = true;
            if (k == 1) {
                count(psite::draft);
            }
        }
    }
    return true;
}

bool prompt_path::check_errors(std::string & err) {
    for (int32_t k = 0; k < 2; ++k) {
        uint32_t w[4] = { 0, 0, 0, 0 };
        if (!cuda_ok(cudaSetDevice(m_->device[k]), "device", err) ||
            !cuda_ok(cudaMemcpy(w, a_[size_t(k)].err, sizeof(w), cudaMemcpyDeviceToHost), "error words", err) ||
            // cleared on the compute stream (non-blocking: a legacy-stream memset is not ordered before the next
            // chunk's kernels), then drained
            !cuda_ok(cudaMemsetAsync(a_[size_t(k)].err, 0, sizeof(w), static_cast<cudaStream_t>(p_[k]->c.stream)),
                     "error words", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(p_[k]->c.stream)), "error words", err)) {
            return false;
        }
        if (w[0] != 0u || w[1] != 0u || w[2] != 0u) {
            char b[160];
            snprintf(b, sizeof(b),
                     "prompt: card %d error words: prompt 0x%x (bit 0 an FP16 range, 25 a message generation, 26 a "
                     "partial's range, 30 a test failure), select %u, experts %u",
                     k, w[0], w[1], w[2]);
            return fail(err, b);
        }
    }
    return true;
}

// ---- TTTS1: the split's crossing ----

namespace {

// the crossing's layout for piece h of c (MUP2; the halves when it names none): S (card 0's), inject of card 0 and 1,
// the FFN message of card 0 and 1
struct crossing_half {
    int64_t s = 0, inj[2] = { 0, 0 }, p[2] = { 0, 0 }, end = 0;
};

crossing_half crossing_at(const prompt_chunk & c, int32_t h, int64_t base) {
    const int64_t n = prompt_piece_rows(c, h);
    crossing_half x;
    x.s      = base;
    x.inj[0] = x.s + n * 4 * D * 4;
    x.inj[1] = x.inj[0] + n * 4 * 4;
    x.p[0]   = x.inj[1] + n * 4 * 4;
    x.p[1]   = x.p[0] + fn::xchg::pmsg_bytes(n * D * 2);
    x.end    = x.p[1] + fn::xchg::pmsg_bytes(n * D * 2);
    return x;
}

}  // namespace

int64_t prompt_path::crossing_bytes(const prompt_chunk & c) {
    int64_t off = 0;
    for (int32_t h = 0; h < prompt_pieces(c); ++h) {
        off = crossing_at(c, h, off).end;
    }
    return off;
}

bool prompt_path::crossing_out(const prompt_chunk & c, int32_t k, uint8_t * dst, std::string & err) {
    if (m_->single || k < 1 || k >= fn::n_layer || dst == nullptr) {
        return fail(err, "prompt crossing out: two cards, k in 1 .. 47 and a buffer");
    }
    if (!prompt_pieces_ok(c)) {
        return fail(err, "prompt crossing out: the chunk's pieces");
    }
    int64_t off = 0;
    for (int32_t h = 0; h < prompt_pieces(c); ++h) {
        const crossing_half x = crossing_at(c, h, off);
        off                   = x.end;
        const int32_t  r0     = prompt_piece_r0(c, h);
        const int64_t  n      = prompt_piece_rows(c, h);
        link &         lp     = links_[lk_p0 + h];
        const uint64_t gen    = fn::xchg::pgen(c.seq, k - 1, 1, h, fn::xchg::pkind::ffn);
        for (int32_t kk = 0; kk < 2; ++kk) {
            card_arena & a = a_[size_t(kk)];
            cudaStream_t s = static_cast<cudaStream_t>(p_[kk]->c.stream);
            if (lp.g_prod[kk] != gen || lp.g_copied[1 - kk] != gen) {
                return fail(err, "prompt crossing out: layer " + std::to_string(k - 1) +
                                     "'s FFN message was not produced and copied");
            }
            if (!cuda_ok(cudaSetDevice(m_->device[kk]), "device", err) ||
                (kk == 0 && !cuda_ok(cudaMemcpyAsync(dst + x.s, a.S + int64_t(r0) * 4 * D, size_t(n) * 4 * D * 4,
                                                     cudaMemcpyDeviceToHost, s),
                                     "crossing S", err)) ||
                !cuda_ok(cudaMemcpyAsync(dst + x.inj[kk], a.inject + int64_t(r0) * 4, size_t(n) * 4 * 4,
                                         cudaMemcpyDeviceToHost, s),
                         "crossing inject", err) ||
                !cuda_ok(cudaMemcpyAsync(dst + x.p[kk], p_msg(kk, true, c, h), size_t(fn::xchg::pmsg_bytes(n * D * 2)),
                                         cudaMemcpyDeviceToHost, s),
                         "crossing partial", err) ||
                // the consumer's release of the receive slot (layer k's phase 0 on one box)
                !cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(lp.copied[1 - kk]), 0), "crossing", err) ||
                !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(lp.freed[kk]), s), "crossing freed", err)) {
                return false;
            }
            lp.g_freed[kk]   = gen;
            lp.any_freed[kk] = true;
        }
    }
    return true;
}

bool prompt_path::crossing_in(const prompt_chunk & c, int32_t k, const uint8_t * src, std::string & err) {
    if (m_->single || k < 1 || k >= fn::n_layer || src == nullptr) {
        return fail(err, "prompt crossing in: two cards, k in 1 .. 47 and a buffer");
    }
    if (!prompt_pieces_ok(c)) {
        return fail(err, "prompt crossing in: the chunk's pieces");
    }
    int64_t off = 0;
    for (int32_t h = 0; h < prompt_pieces(c); ++h) {
        const crossing_half x = crossing_at(c, h, off);
        off                   = x.end;
        const int32_t  r0     = prompt_piece_r0(c, h);
        const int64_t  n      = prompt_piece_rows(c, h);
        link &         lp     = links_[lk_p0 + h];
        const uint64_t gen    = fn::xchg::pgen(c.seq, k - 1, 1, h, fn::xchg::pkind::ffn);
        for (int32_t kk = 0; kk < 2; ++kk) {
            card_arena & a = a_[size_t(kk)];
            cudaStream_t s = static_cast<cudaStream_t>(p_[kk]->c.stream);
            if (!cuda_ok(cudaSetDevice(m_->device[kk]), "device", err) ||
                !cuda_ok(cudaMemcpyAsync(a.S + int64_t(r0) * 4 * D, src + x.s, size_t(n) * 4 * D * 4,
                                         cudaMemcpyHostToDevice, s),
                         "crossing S", err) ||
                !cuda_ok(cudaMemcpyAsync(a.inject + int64_t(r0) * 4, src + x.inj[kk], size_t(n) * 4 * 4,
                                         cudaMemcpyHostToDevice, s),
                         "crossing inject", err) ||
                !cuda_ok(cudaMemcpyAsync(p_msg(kk, true, c, h), src + x.p[kk], size_t(fn::xchg::pmsg_bytes(n * D * 2)),
                                         cudaMemcpyHostToDevice, s),
                         "crossing partial", err) ||
                !cuda_ok(cudaMemcpyAsync(p_msg(kk, false, c, h), src + x.p[1 - kk],
                                         size_t(fn::xchg::pmsg_bytes(n * D * 2)), cudaMemcpyHostToDevice, s),
                         "crossing peer partial", err) ||
                // the exchange's copy stands done: card kk's message is in the peer's receive slot (the bookkeeping
                // layer k's phase 0 checks; its data is ordered on each card's own stream above)
                !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(lp.prod[kk]), s), "crossing produced", err) ||
                !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(lp.copied[kk]), s), "crossing copied", err)) {
                return false;
            }
            lp.g_prod[kk]     = gen;
            lp.g_copied[kk]   = gen;
            lp.any_copied[kk] = true;
        }
    }
    return true;
}

bool prompt_path::publish(const prompt_chunk & c, std::string & err) {
    const int32_t last = c.rows - 1;
    for (int32_t k = 0; k < 2; ++k) {
        card_arena &  a   = a_[size_t(k)];
        cudaStream_t s = static_cast<cudaStream_t>(p_[k]->c.stream);
        const float * row = a.S + int64_t(last) * 4 * D;
        if (!cuda_ok(cudaSetDevice(m_->device[k]), "device", err) ||
            ((!m_->single || k == 0) &&
             !cuda_ok(cudaMemcpyAsync(a.pred, row, size_t(4) * D * 4, cudaMemcpyDeviceToDevice, s), "pred", err)) ||
            (d_[k] != nullptr &&
             !cuda_ok(cudaMemcpy2DAsync(d_[k]->carry_prev, size_t(F) * 4, row + ptrdiff_t(k) * F, size_t(D) * 4,
                                        size_t(F) * 4, 4, cudaMemcpyDeviceToDevice, s),
                      "retained carry", err))) {
            return false;
        }
    }
    return true;
}

bool prompt_path::head_input(const prompt_chunk & c, std::string & err) {
    const int32_t last = c.rows - 1;
    for (int32_t k = 0; k < 2; ++k) {
        card_arena &  a   = a_[size_t(k)];
        pass_card &   p   = *p_[k];
        cudaStream_t  s   = static_cast<cudaStream_t>(p.c.stream);
        const float * row = a.S + int64_t(last) * 4 * D;
        if (!cuda_ok(cudaSetDevice(m_->device[k]), "device", err) ||
            !cuda_ok(cudaMemcpy2DAsync(p.c.ws.x[1], size_t(F) * 4, row + ptrdiff_t(k) * F, size_t(D) * 4, size_t(F) * 4,
                                       4, cudaMemcpyDeviceToDevice, s),
                     "head input", err) ||
            !cuda_ok(cudaMemsetAsync(p.c.ws.inject[1], 0, size_t(16), s), "head input", err) ||
            !cuda_ok(cudaMemsetAsync(p.c.ws.out[1], 0, size_t(F) * 4, s), "head input", err)) {
            return false;
        }
    }
    count(psite::head);
    return true;
}

bool prompt_path::dump_hc(const prompt_chunk & c, int32_t layer, int32_t which, int32_t h, std::string & err) {
    if (dump_.row_stride <= 0 || std::find(dump_.layers.begin(), dump_.layers.end(), layer) == dump_.layers.end() ||
        std::find(dump_.chunks.begin(), dump_.chunks.end(), c.index) == dump_.chunks.end()) {
        return true;
    }
    // which 0/1: the attention/FFN call's input streams and low-rank coefficients (after mid); 2/3: its mixed output
    card_arena &  a  = a_[0];
    cudaStream_t  s  = static_cast<cudaStream_t>(p_[0]->c.stream);
    const int32_t r0 = prompt_piece_r0(c, h);
    const int32_t n  = prompt_piece_rows(c, h);
    const int32_t m  = (n + dump_.row_stride - 1) / dump_.row_stride;
    bump          b{ a.scratch };
    (void) b.take<float>(int64_t(n) * fn::hc::hcp_mom * 4);
    const float *      low = b.take<float>(int64_t(n) * 320 * 4);
    std::vector<float> host;
    const auto         grab = [&](const float * src, int64_t width, int64_t ld) {
        host.assign(size_t(m) * width, 0.0f);
        return cuda_ok(cudaStreamSynchronize(s), "dump", err) &&
               cuda_ok(cudaMemcpy2D(host.data(), size_t(width) * 4, src, size_t(ld) * dump_.row_stride * 4,
                                    size_t(width) * 4, size_t(m), cudaMemcpyDeviceToHost),
                       "dump", err);
    };
    const auto put = [&](const std::string & name) {
        const std::string path = dump_.dir + "/" + name;
        FILE *            f    = fopen(path.c_str(), "wb");
        if (f == nullptr || fwrite(host.data(), 4, host.size(), f) != host.size()) {
            if (f != nullptr) {
                fclose(f);
            }
            return fail(err, "prompt: cannot write " + path);
        }
        fclose(f);
        return true;
    };
    const std::string tag = "c" + std::to_string(c.index) + "-L" + std::to_string(layer) + "-" +
                            ((which & 1) == 0 ? "attn" : "ffn") + "-h" + std::to_string(h);
    if (which < 2) {
        return grab(a.S + int64_t(r0) * 4 * D, int64_t(4) * D, int64_t(4) * D) && put(tag + "-x.f32") &&
               grab(low, 320, 320) && put(tag + "-low.f32");
    }
    return grab(a.mixed, D, D) && put(tag + "-mixed.f32");
}

}  // namespace fn::model
