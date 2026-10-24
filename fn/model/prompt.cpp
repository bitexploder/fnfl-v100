#include "fn/model/prompt.h"

#include "fn/kernels/dense/dense_prompt.h"
#include "fn/kernels/draft/draft_prompt.h"
#include "fn/kernels/embed/embed.h"
#include "fn/kernels/experts/experts.h"
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
#include "fn/model/kv_tier.h"
#include "fn/model/shapes.h"
#include "fn/xchg/prompt_xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
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

// the slot sizes (header and payload); ME3: for a layout of rc chunk rows in halves of rh (prompt_path::layout)
constexpr int64_t p_slot(int64_t rh) {
    return 256 + rh * D * 2;
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
    const int64_t ffn_rt = al(int64_t(H) * NE * 4) + al(int64_t(H) * 8 * 4) + 2 * al(int64_t(H) * 4) +
                           al(int64_t(H) * NU * 4) * 2 + al(int64_t(513) * 4) + 2 * al(int64_t(H) * NU * 4) +
                           al(int64_t(fn::experts::ep_max_tasks) * 16) + al(int64_t(8) * 4) + al(int64_t(513) * 4);
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
    uint8_t *p_send[2] = {}, *p_recv[2] = {};
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

enum link_id : int32_t { lk_p0, lk_p1, lk_emb, lk_sel0, lk_sel1, lk_mom, lk_dmix, n_links };

int64_t arena_bytes_a(int64_t rc, int64_t rh, bool single, int32_t card) {
    if (!single) {
        int64_t n = 0;
        n += al(rc * 4 * D * 4) + al(rc * D * 4) + al(rc * 4) + al(rc * 4 * 4);
        n += al(rh * D * 4) + al(rh * D * 2) + al(rh * D * 4);
        n +=
            4 * al(p_slot(rh)) + 2 * al(emb_slot(rc)) + 4 * al(sel_slot) + 2 * al(mom_slot(rh)) + 2 * al(dmix_slot(rh));
        n += al(int64_t(4) * D * 4) + 2 * al(int64_t(4) * D * 9 * 4) + al(ple_hist_card * 4) + al(64);
        n += al(family_bytes(rh)) + al(conv_bytes());
        return n;
    }
    int64_t n = 0;
    if (card == 0) {
        n += al(rc * 4 * D * 4) + al(rc * D * 4) + al(rc * 4) + al(rc * 4 * 4);
        n += al(rh * D * 2);
    }
    n += 2 * al(p_slot(rh)) + 2 * al(sel_slot) + al(mom_slot(rh)) + al(dmix_slot(rh));
    if (card == 0) {
        n += al(int64_t(4) * D * 4) + 2 * al(int64_t(4) * D * 9 * 4);
    }
    n += al(64) + al(family_bytes_single(rc, rh, card)) + al(conv_bytes(true, card));
    return n;
}

} // namespace

int64_t prompt_arena_bytes(int32_t chunk_rows, int32_t half_rows, bool single, int32_t card) {
    return arena_bytes_a(chunk_rows, half_rows, single, card);
}

int64_t prompt_replica_bytes(bool single) {
    const int64_t norms = al(int64_t(D) * 4) + al(int64_t(4) * D * 4);
    if (single) {
        return norms;
    }
    return int64_t(2) * fn::n_layer * al(hc_down_bytes + hc_up_bytes + hc_gamma_bytes) + al(ple_key_bytes) +
           al(ple_value_bytes) + 3 * al(ple_norm_bytes) + al(ple_conv_bytes) + norms;
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

prompt_path::~prompt_path() {
    for (int32_t c = 0; c < 2 && m_ != nullptr; ++c) {
        cudaSetDevice(m_->device[c]);
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
    for (int32_t h = 0; h < 2; ++h) {
        a.p_send[h] = b.take<uint8_t>(p_slot(rh));
        if (!single) {
            a.p_recv[h] = b.take<uint8_t>(p_slot(rh));
        }
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
            for (int32_t h = 0; h < 2; ++h) {
                a.p_recv[h] = p.p_send[h];
                a.sel_recv[h] = p.sel_send[h];
            }
            a.emb_recv = p.emb_send;
            a.mom_recv = p.mom_send;
            a.dmix_recv = p.dmix_send;
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        card_arena & a = a_[size_t(c)];
        links_[lk_p0].send[c] = a.p_send[0];
        links_[lk_p0].recv[c]   = a.p_recv[0];
        links_[lk_p1].send[c]   = a.p_send[1];
        links_[lk_p1].recv[c]   = a.p_recv[1];
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
                hh.down[c]                  = own.w_down;
                hh.up[c]                    = own.w_up;
                hh.gamma[c]                 = own.gamma;
                if (one) {
                    const hc_module & pw = sub == 0 ? p[1 - c]->w[l].hc_attn : p[1 - c]->w[l].hc_ffn;
                    hh.down[1 - c]       = pw.w_down;
                    hh.up[1 - c]         = pw.w_up;
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
        a.ple_conv[c]      = p[c]->ple_w.conv;
        if (one) {
            const fn::ple::weights & pw = p[1 - c]->ple_w;
            a.ple_key[1 - c]            = reinterpret_cast<const uint16_t *>(pw.key);
            a.ple_value[1 - c]          = reinterpret_cast<const uint16_t *>(pw.value);
            a.ple_nk[1 - c]             = pw.norm_key;
            a.ple_nq[1 - c]             = pw.norm_query;
            a.ple_nc[1 - c]             = pw.norm_conv;
            a.ple_conv[1 - c]           = pw.conv;
        } else {
            a.ple_key[1 - c]   = r.take<const uint16_t>(ple_key_bytes);
            a.ple_value[1 - c] = r.take<const uint16_t>(ple_value_bytes);
            a.ple_nk[1 - c]    = r.take<const float>(ple_norm_bytes);
            a.ple_nq[1 - c]    = r.take<const float>(ple_norm_bytes);
            a.ple_nc[1 - c]    = r.take<const float>(ple_norm_bytes);
            a.ple_conv[1 - c]  = r.take<const uint16_t>(ple_conv_bytes);
        }
        a.enorm = r.take<float>(int64_t(D) * 4);
        a.hnorm = r.take<float>(int64_t(4) * D * 4);
        if (r.off > prompt_replica_bytes(one)) {
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
    return fill_replicas(err);
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
            for (int32_t sub = 0; sub < 2; ++sub) {
                const card_arena::hc_halves & hh  = sub == 0 ? a.hc_attn[size_t(l)] : a.hc_ffn[size_t(l)];
                const hc_module &             src = sub == 0 ? p_[pc]->w[l].hc_attn : p_[pc]->w[l].hc_ffn;
                if (!peer(hh.down[pc], src.w_down, hc_down_bytes, "HC down replica") ||
                    !peer(hh.up[pc], src.w_up, hc_up_bytes, "HC up replica") ||
                    !peer(hh.gamma[pc], src.gamma, hc_gamma_bytes, "HC gamma replica")) {
                    return false;
                }
            }
        }
        const fn::ple::weights & pw = p_[pc]->ple_w;
        if (!m_->single && (!peer(a.ple_key[pc], pw.key, ple_key_bytes, "PLE key replica") ||
            !peer(a.ple_value[pc], pw.value, ple_value_bytes, "PLE value replica") ||
            !peer(a.ple_nk[pc], pw.norm_key, ple_norm_bytes, "PLE norm replica") ||
            !peer(a.ple_nq[pc], pw.norm_query, ple_norm_bytes, "PLE norm replica") ||
            !peer(a.ple_nc[pc], pw.norm_conv, ple_norm_bytes, "PLE norm replica") ||
            !peer(a.ple_conv[pc], pw.conv, ple_conv_bytes, "PLE conv replica"))) {
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
    return static_cast<fn::experts::fmt>(ef) == fn::experts::fmt::q2_0 ? fn::dense::fmt::q2_0 : fn::dense::fmt::iq4_nl;
}

fn::dense::pd_act act1(const uint16_t * x, int64_t ld) {
    fn::dense::pd_act a;
    a.a0      = x;
    a.ld0     = ld;
    a.k_split = static_cast<int32_t>(ld);
    return a;
}

}  // namespace

bool prompt_path::chunk(const prompt_chunk & c, std::string & err) {
    return chunk_begin(c, err) && chunk_layers(c, 0, fn::n_layer, err) && chunk_end(c, err);
}

bool prompt_path::chunk_begin(const prompt_chunk & c, std::string & err) {
    if (c.rows < 1 || c.rows > rc_ || c.half[0] < 1 || c.half[0] > rh_ || c.half[1] < 0 || c.half[1] > rh_ ||
        c.half[0] + c.half[1] != c.rows || c.tok == nullptr || c.e[0] == nullptr || c.e[1] == nullptr) {
        return fail(err, "prompt: a chunk outside the layout's " + std::to_string(rc_) + " rows in halves of at most " +
                             std::to_string(rh_));
    }
    // the tokens to both cards (a pinned slot a chunk parity: the slot two chunks back has been copied, the chunk
    // end's join having passed)
    for (int32_t k = 0; k < 2; ++k) {
        card_arena & a = a_[size_t(k)];
        if (!cuda_ok(cudaSetDevice(m_->device[k]), "device", err)) {
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
            if (phase == 0) {
                if (!m_->single && lk.any_copied[k] &&
                    !cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(lk.copied[k]), 0), "emb reuse", err)) {
                    return false;
                }
                if (!fn::embed::embed_rows(ea, s) ||
                    (!m_->single &&
                     (!fn::xchg::pstamp(a.emb_send, gen, c.rows, fn::xchg::pkind::emb, s) ||
                      !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(lk.prod[k]), s), "emb produced", err)))) {
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
    const int32_t nh = c.half[1] > 0 ? 2 : 1;
    // the layers, sublayer-major in halves
    for (int32_t l = lo; l < hi; ++l) {
        for (int32_t sub = 0; sub < 2; ++sub) {
            for (int32_t h = 0; h < nh; ++h) {
                for (int32_t phase = 0; phase < 4; ++phase) {
                    if (!layer_half(c, l, sub, h, phase, err)) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool prompt_path::chunk_end(const prompt_chunk & c, std::string & err) {
    const int32_t nh = c.half[1] > 0 ? 2 : 1;
    // the target's end: layer 47's FFN exchange into the final streams (phase 0 of a layer 48 attention sublayer)
    for (int32_t h = 0; h < nh; ++h) {
        if (!layer_half(c, fn::n_layer, 0, h, 0, err)) {
            return false;
        }
    }
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

// One phase of sublayer sub of layer l for half h, both cards (card 0, then card 1):
//   0: the previous sublayer's exchange of this half (receive, combine into S); at layer 1's attention, PLE; HC; the
//      body's first part (products; on a QSA layer's index owner the index and selection, the SEL message produced)
//   1: the SEL copy (QSA layers, the owner)
//   2: the body's rest (QSA: the peer acquires SEL; K/V; attention; wo), the partial packed, the P message produced
//   3: the P copy
// l = 48 runs phase 0's receive and combine only (the target's end).
bool prompt_path::layer_half(const prompt_chunk & c,
                             int32_t              l,
                             int32_t              sub,
                             int32_t              h,
                             int32_t              phase,
                             std::string &        err) {
    const int32_t  r0       = h == 0 ? 0 : c.half[0];
    const int32_t  n        = c.half[h];
    const int64_t  pos0     = c.start + r0;
    const bool     end      = l == fn::n_layer;
    const bool     gdn      = !end && l % fn::full_attn_interval != fn::full_attn_interval - 1;
    link &         lp       = links_[h == 0 ? lk_p0 : lk_p1];
    link &         ls       = links_[h == 0 ? lk_sel0 : lk_sel1];
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
                cudaStream_t cs = static_cast<cudaStream_t>(copy_[k]);
                if (!cuda_ok(cudaSetDevice(m_->device[k]), "device", err) ||
                    !cuda_ok(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(lk.prod[k]), 0), "copy wait", err) ||
                    (lk.any_freed[1 - k] &&
                     !cuda_ok(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(lk.freed[1 - k]), 0), "copy free",
                              err)) ||
                    !cuda_ok(cudaMemcpyPeerAsync(lk.recv[1 - k], m_->device[1 - k], lk.send[k], m_->device[k],
                                                 size_t(phase == 1 ? sel_used : bytes), cs),
                             "copy", err) ||
                    // SEL (PB3): the used slabs' bodies above, then their offsets (the payload's two regions)
                    (phase == 1 &&
                     !cuda_ok(cudaMemcpyPeerAsync(lk.recv[1 - k] + sel_offs_at, m_->device[1 - k],
                                                  lk.send[k] + sel_offs_at, m_->device[k], size_t(sel_offs_used), cs),
                              "copy", err)) ||
                    !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(lk.copied[k]), cs), "copied", err)) {
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
            const auto freed = [&]() {
                if (!m_->single && !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(lp.freed[k]), s), "freed", err)) {
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
                if (!m_->single &&
                    !cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(lp.copied[1 - k]), 0), "recv", err)) {
                    return false;
                }
                if (fold) {
                    cb.card    = k;
                    cb.own     = a.p_send[h];
                    cb.peer    = a.p_recv[h];
                    cb.gen     = gen_prev;
                    cb.inject  = inj;
                    cb.streams = S;
                } else if (!fn::xchg::pcombine(n, k, a.p_send[h], a.p_recv[h], gen_prev, inj, S, a.err, s) ||
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
                pa.v_out    = h + 1;
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
            }
            // 0c: HC
            if (!m_->single || k == 0) {
                const card_arena::hc_halves & hh = sub == 0 ? a.hc_attn[size_t(l)] : a.hc_ffn[size_t(l)];
                bump b{sc};
                fn::hc::hcp_args ha;
                ha.w.form = fn::hc::hcp_form::replicated;
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
                // the coefficients to FP16 at the sublayer's first half (the second half reads them)
                if ((h == 0 && !fn::hc::hct_convert(ha.w, a.hct, s)) ||
                    !fn::hc::hct_mix(ha, cb, a.hct, b.take<uint8_t>(fn::hc::hct_scratch_bytes(n)), s) ||
                    (fold && !freed())) {
                    return err.empty() ? fail(err, "prompt: HC") : false;
                }
                count(sub == 0 ? psite::hc_attn : psite::hc_ffn);
                if (k == 0 && (!dump_hc(c, l, sub, h, err) || !dump_hc(c, l, sub + 2, h, err))) {
                    return false;
                }
            }
            // 0d: the body's first part. PB3: the sublayer's producer (GDN out, the experts' sum, QSA wo) writes the P
            // message's FP16 payload itself, so the send slot's previous message has been copied out before the body
            if (!m_->single && lp.any_copied[k] &&
                !cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(lp.copied[k]), 0), "slot reuse", err)) {
                return false;
            }
            uint16_t * const pay = reinterpret_cast<uint16_t *>(a.p_send[h] + fn::xchg::pmsg_payload_at);
            const fn::dense::pd_act mx   = act1(a.mixed16, D);
            // PB3: the sublayer's dense weights to FP16 in the conversion region at its first half (fused matrices
            // contiguous), then tensor products with one output segment a matrix
            const auto              to16 = [&](const dense_weight & dw, int64_t at) {
                fn::dense::pd_args pa;
                pa.f     = static_cast<fn::dense::fmt>(dw.fmt);
                pa.lay   = fn::dense::pd_layout::dp2;
                pa.k     = static_cast<int32_t>(dw.k);
                pa.n     = static_cast<int32_t>(dw.n_out);
                pa.w     = dw.w;
                pa.grids = p.c.dense_grids;
                return h != 0 || fn::dense::pd_convert(pa, a.w16 + at, s);
            };
            // segs: up to three output segments; cs: the BF16 conversions' column scales (at cs_at), or -1; o16: the
            // P payload instead of the one segment
            const auto tprod = [&](int64_t at, int32_t nn, int32_t kk, const fn::dense::pd_act & x,
                                   const fn::dense::pt_seg * segs, int32_t n_seg, int64_t cs_at,
                                   uint16_t * o16 = nullptr, int64_t l16 = D, uint32_t e16 = fn::xchg::err_prange) {
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
                t.out16 = o16;
                t.ld16 = l16;
                t.err = a.err;
                t.err_bit = e16;
                return fn::dense::pd_tc(t, s);
            };
            // BF16 rows of K = 2,560 to FP16 at at, their column scales at s_at (at the sublayer's first half)
            const auto to16b = [&](const uint16_t * bw, int32_t rows_w, int64_t at, int64_t s_at) {
                return h != 0 || fn::dense::pd_convert_bf16(bw, rows_w, D, a.w16 + at,
                                                            reinterpret_cast<float *>(a.w16 + s_at), s);
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
                if (!to16(w.qkv, 0) || !to16(w.z, int64_t(GC) * D) || !to16(w.ssm_out, gdn_out_at) ||
                    !to16b(w.alpha, 24, gdn_ab_at, gdn_ab_s_at) ||
                    !to16b(w.beta, 24, gdn_ab_at + int64_t(24) * D, gdn_ab_s_at + 48) ||
                    !tprod(0, GC + GV, D, mx, in, 1, -1, qz16, GC + GV, 1u) ||
                    !tprod(gdn_ab_at, 48, D, mx, ab, 2, gdn_ab_s_at)) {
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
                ga.rec      = st.rec;    // ME11: half 0 replays the selector's records
                ga.gsel     = p.c.gsel;
                ga.cur      = p.c.cur;
                ga.v_in     = h;
                ga.v_out    = h + 1;
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
                count(psite::gdn_core);
                const fn::dense::pt_seg po[1] = {{nullptr, D, D}};
                if (!tprod(gdn_out_at, D, GV, act1(gated16, GV), po, 1, -1, pay)) {
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
                if (!to16(w.wq, 0) || !to16(w.wk, int64_t(QF) * D) || !to16(w.wv, int64_t(QF + 256) * D) ||
                    !to16(w.wo, qsa_out_at) || !tprod(0, QF + 512, D, mx, in, 3, -1)) {
                    return fail(err, "prompt: QSA input");
                }
                const fn::dense::pt_seg ix[2] = {{iq, 512, 512}, {ik, 128, 640}};
                if (k == w.index_owner && (!to16b(w.idx_q, 512, qsa_idx_at, qsa_idx_s_at) ||
                                           !to16b(w.idx_k, 128, qsa_idx_at + int64_t(512) * D, qsa_idx_s_at + 1024) ||
                                           !tprod(qsa_idx_at, 640, D, mx, ix, 2, qsa_idx_s_at))) {
                    return fail(err, "prompt: QSA index input");
                }
                count(psite::qsa_input);
                if (k == w.index_owner) {
                    if (!m_->single && ls.any_copied[k] &&
                        !cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(ls.copied[k]), 0), "sel reuse", err)) {
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
                    fn::qsa::qsp_index_args ia;
                    ia.summaries = st.idx_summaries;
                    ia.iq        = iq;
                    ia.q_norm    = w.idx_q_norm;
                    ia.rope      = fn::qsa::model_rope();
                    ia.rows      = n;
                    ia.pos0      = static_cast<int32_t>(pos0);
                    ia.max_cells = st.n_cells;
                    ia.ws        = fn::qsa::prompt_carve(wsb, st.n_cells);
                    ia.ws.error  = a.err + 1;
                    ia.body      = reinterpret_cast<uint32_t *>(a.sel_send[h] + fn::xchg::pmsg_payload_at);
                    ia.offsets   = reinterpret_cast<int32_t *>(ia.body + fn::qsa::prompt_body_words);
                    if (!fn::qsa::qsp_keys(n, static_cast<int32_t>(pos0), ik, keys, st.n_cells, s) ||
                        (ua.count > 0 && !fn::qsa::summary_update(ua, s)) ||
                        (kt != nullptr && !kt->write_back_keys(k, st.kv_attn, pos0, n, s, err)) ||
                        !fn::qsa::qsp_select(ia, s) ||
                        !fn::xchg::pstamp(a.sel_send[h], gen_sel, n, fn::xchg::pkind::sel, s) ||
                        (!m_->single &&
                         !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ls.prod[k]), s), "sel produced", err))) {
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
                    if ((h == 0 && (!cuda_ok(cudaMemsetAsync(a.w16 + ffn_router_at + int64_t(NE + 1) * D, 0,
                                                             size_t(7) * D * 2, s),
                                             "gate pad", err) ||
                                    !cuda_ok(cudaMemsetAsync(rsc + NE + 1, 0, size_t(7) * 4, s), "gate pad", err))) ||
                        !to16b(w.router, NE, ffn_router_at, ffn_router_s_at) ||
                        !to16b(w.sh_gate_w, 1, ffn_router_at + int64_t(NE) * D, ffn_router_s_at + 2 * NE) ||
                        !tprod(ffn_router_at, NE + 8, D, mx, rs, 2, ffn_router_s_at) ||
                        !fn::dense::pd_route(n, logits, sgl, ids, wts, sg, s, 8)) {
                        return fail(err, "prompt: router");
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
                    const int32_t * const rid = c.half[1] == 0 ? ids : nullptr;
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
                if ((own && !fn::experts::ep_group(n, fn::experts::ep_task_rows(gf), ids, rt, a.err + 2, s)) ||
                    (own && skip != nullptr && !fn::experts::es_drop(rt, skip, s)) ||
                    (ring ? (own && !fn::experts::ep_task_offsets(rt, toff, s))
                          : !fn::experts::ep_gate_up(gf, n, gu, s))) {
                    return fail(err, "prompt: experts gate/up");
                }
                // the shared expert: gate and up (DP2's packing), the GLU, its down (EX1's) scaled by sg
                if (w.sh_gate.n_out != 320 || w.sh_up.n_out != 320) {
                    return fail(err, "prompt: shared expert shapes");
                }
                const fn::dense::pt_seg gu2[2] = {{sh_g, 320, 320}, {sh_u, 320, 640}};
                if (!to16(w.sh_gate, 0) || !to16(w.sh_up, int64_t(320) * D) || !tprod(0, 640, D, mx, gu2, 2, -1) ||
                    !fn::dense::pd_glu(n, 320, sh_g, sh_u, 320, sh_glu, 320, s)) {
                    return fail(err, "prompt: shared expert");
                }
                {
                    // the shared down (EX1's layout), scaled by sg after the sum
                    fn::dense::pd_args pa;
                    pa.f     = dense_of(w.ex_sh_down_fmt);
                    pa.lay   = fn::dense::pd_layout::ex1_down;
                    pa.k     = 320;
                    pa.n     = D;
                    pa.w     = w.ex_sh_down;
                    pa.grids = p.c.dense_grids;
                    fn::dense::pt_args t;
                    t.w         = a.w16 + ffn_down_at;
                    t.n         = D;
                    t.k         = 320;
                    t.rows      = n;
                    t.x         = act1(sh_glu, 320);
                    t.seg[0]    = { shared, D, D };
                    t.row_scale = sg;
                    if ((h == 0 && !fn::dense::pd_convert(pa, a.w16 + ffn_down_at, s)) || !fn::dense::pd_tc(t, s)) {
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
                    if (!fn::experts::ep_down(df, n, dn, s) || !fn::experts::ep_sum(n, sa, s)) {
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
                const uint8_t * sel   = owner ? a.sel_send[h] : a.sel_recv[h];
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
                    if ((!m_->single && !cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(ls.copied[1 - k]), 0),
                                                 "sel recv", err)) ||
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
                aa.max_cells = st.n_cells;
                aa.out16     = att16;
                aa.err       = a.err;
                if (!fn::qsa::qsp_attend(aa, s)) {
                    return fail(err, "prompt: QSA attention");
                }
                if (!owner) {
                    if (!m_->single &&
                        !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ls.freed[k]), s), "sel freed", err)) {
                        return false;
                    }
                    ls.g_freed[k] = gen_sel;
                    ls.any_freed[k] = true;
                }
                count(psite::qsa_attend);
                // wo: converted at phase 0 of the sublayer's first half
                fn::dense::pt_args t;
                t.w      = a.w16 + qsa_out_at;
                t.n      = D;
                t.k      = GV;
                t.rows   = n;
                t.x      = act1(att16, GV);
                t.seg[0] = {nullptr, D, D};
                t.out16 = reinterpret_cast<uint16_t *>(a.p_send[h] + fn::xchg::pmsg_payload_at);
                t.ld16 = D;
                t.err = a.err;
                t.err_bit = fn::xchg::err_prange;
                if (!fn::dense::pd_tc(t, s)) {
                    return fail(err, "prompt: QSA out");
                }
                count(psite::qsa_out);
            }
            // the payload written by the sublayer's producer (phase 0 waited for the slot's previous copy)
            if (!fn::xchg::pseal(n, a.p_send[h], gen, s) ||
                (!m_->single && !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(lp.prod[k]), s), "produced", err))) {
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
        hist_cur_ ^= 1;  // the next half's PLE reads this half's history
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
        const auto to16d = [&](const dense_weight & dw, int64_t at) {
            fn::dense::pd_args pa;
            pa.f = static_cast<fn::dense::fmt>(dw.fmt);
            pa.lay = fn::dense::pd_layout::dp2;
            pa.k = static_cast<int32_t>(dw.k);
            pa.n = static_cast<int32_t>(dw.n_out);
            pa.w = dw.w;
            pa.grids = p.c.dense_grids;
            return fn::dense::pd_convert(pa, a.w16 + at, s);
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
            bool cv = h != 0 || (to16d(d.w.eh, 0) && to16d(d.w.wk, draft_wk_at) && to16d(d.w.wv, draft_wv_at));
            if (h == 0 && d.sparse) {
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
            if (!cv || !fn::draft::dcp_norm(na, s) || !fn::dense::pd_tc(pt, s) || !fn::hc::hcp_down(ha, s) ||
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
            const auto q8 = [&](int64_t at, float * out) { // wk or wv, converted at phase 0 of the first half
                fn::dense::pt_args pt;
                pt.w = a.w16 + at;
                pt.n = 256;
                pt.k = D;
                pt.rows = n;
                pt.x = act1(mixed16, D);
                pt.seg[0] = {out, 256, 256};
                pt.n_seg = 1;
                return fn::dense::pd_tc(pt, s);
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
            if ((!m_->single &&
                 !cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(ld.copied[1 - k]), 0), "dmix recv", err)) ||
                !fn::xchg::pcheck(a.dmix_recv, gmix, a.err, s) ||
                !fn::draft::dcp_mixed(n, k, reinterpret_cast<const float *>(a.dmix_send + fn::xchg::pmsg_payload_at),
                                      reinterpret_cast<const float *>(a.dmix_recv + fn::xchg::pmsg_payload_at), mixed16,
                                      a.err, s) ||
                (!m_->single &&
                 !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ld.freed[k]), s), "dmix freed", err)) ||
                !q8(draft_wk_at, kk) || !q8(draft_wv_at, vv) || !fn::draft::dcp_prepare(kp, s)) {
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
                if (!fn::dense::pd_tc(pt, s) ||
                    !fn::qsa::qsp_keys(n, static_cast<int32_t>(pos0), ik, keys, d.max_cells, s) ||
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
    const int32_t r0 = h == 0 ? 0 : c.half[0];
    const int32_t n  = c.half[h];
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
