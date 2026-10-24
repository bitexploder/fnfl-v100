#pragma once

// SR1: the slot's versioned state on the host (FC-SLOT1, FC-ST1, FC-ST2, FC-ST3 within the last step). One slot on two
// cards. Host code only.
//
// The ledger (FC-SLOT1): L (tok.size()), N, U, pending, retained, prompt_len, O and A, the summaries' valid blocks and
// the committed slot cur (fn/kernels/round/version.h), mirrored from both cards' round state.
//
// The last step's record (FC-SLOT1's pre_step): cur_pre (the committed slot before the step), N, L, O and the pending
// and retained state before it, the k it committed and the rows it keeps now. The pre-step contents are not copied:
// they stay in slot cur_pre because a round writes only the four slots other than its cur, and the next round, which
// overwrites them, starts only after this step is acknowledged (RS1 section 3: acknowledgment is the next step's start).
// So the record is valid from a step's completion until the next round starts.
//
// The one primitive, cut (truncate within the last step to j kept rows, 0 <= j <= kept, and the ledger to n_keep
// tokens), serves SP1's commit of a + 1 rows, FC-DEC3's lowered a on an EOS or budget cut, SV1's truncate within the
// last step, and the restore after a recoverable error on either card (FC-RD2). The retained row (FC-PF2) follows the
// cut: a step that ran the head has overwritten the logits, so a cut to j = 0 or to 1 <= j < k leaves nothing retained
// (the bootstrap refuses until the last kept token is processed again); a cut of a bootstrap step restores the retained
// row, whose logits the bootstrap only read, so a cancelled bootstrap repeats the same decision. It sets cur <- version_slot(cur_pre, j)
// on both cards (ME11: and the GDN selector, gdn_pre for j = 0, else gdn_after_round(gdn_pre, j); version.h) and
// N <- N_before + j: the KV and raw index-key valid lengths are N (no data moves; cells below
// N_before were not written by the step), and the summaries are cut with invalidate_from(N / 4). A cut block is
// recomputed before it is scored again: the round form of summary_update pools a block in the launch whose rows write
// its last cell, before select scores it (fn/kernels/qsa/qsa.h), and no row sees a block before its last cell is
// written. A cut below N_before is a checkpoint restore and re-prefill (FC-ST3; PF1, SV1), not this primitive.

#include "fn/kernels/qsa/qsa.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fn::runtime {

// FC-SLOT1, one slot
struct ledger {
    std::vector<int32_t> tok; // L = tok.size()
    int64_t n = 0;            // processed
    int64_t u = 0;            // unprocessed, tracked apart from L - N for the check
    int32_t pending = -1;
    bool retained = false; // the last prompt row's logits are on both cards (FC-PF2)
    int32_t retained_row = -1;
    int64_t prompt_len = 0;
    int64_t o = 0, a = 0;
    fn::qsa::summary_index summaries; // blocks 0 .. n_valid - 1 hold summaries of committed cells (n_valid = N / 4)
    uint32_t cur = 0;                 // the committed slot on both cards (SR1)
    uint32_t gdn = 0;                 // ME11: the GDN selector on both cards (fn::round::gdn_word)
    int64_t checks = 0, failures = 0;
};

// the last step, while its versions are intact (FC-SLOT1's pre_step)
struct step_record {
    bool valid = false;
    uint32_t cur_pre = 0; // the committed slot before the step
    uint32_t gdn_pre = 0; // ME11: the GDN selector before the step (its bank and set the step did not write)
    int64_t n_before = 0;
    int64_t l_before = 0;
    int64_t o_before = 0; // outputs before the step (all acknowledged when it started)
    int32_t pending_before = -1;
    bool retained_before = false;
    int32_t retained_row_before = -1;
    int32_t k = 0;     // versions the step committed
    int32_t kept = 0;  // rows it keeps now: k, or j after a cut
    bool head = false; // the step ran the target head (every kind but bootstrap), overwriting the logits
    bool     prompt =
        false;  // PB1: a bulk prompt chunk (its versions are its halves, not rows: no cut within it but j = 0)
};

class slot_versions {
  public:
    // card r's device, stream, committed-slot word and GDN selector word (fn::round::state::cur and gdn, or a test's)
    void bind(int32_t card, int32_t device, void * stream, uint32_t * cur_dev, uint32_t * gdn_dev);

    // cur <- 0 and the GDN selector <- 0 on both cards, the ledger emptied, no step; synchronous
    bool reset(ledger & l, std::string & err);

    // a round is about to start: the previous step is acknowledged (A <- O) and its pre-step slot may be reused, so
    // the record is dropped; the new step's pre-step state is recorded. head: the round runs the target head, which
    // overwrites the logits the retained row points into (every kind but bootstrap)
    void begin_step(ledger & l, bool head);

    // the round completed with every error word clear on both cards: each card's commit moved cur by k. Checks both
    // cards' reported cur against version_slot(cur, k) and (ME11) their GDN selectors against gdn_after_round(gdn, k)
    // (unchanged for k = 0), then records k; the caller then updates N, U and the ledger and calls settle.
    bool committed(ledger & l, int32_t k, const uint32_t reported[2], const uint32_t reported_gdn[2],
                   std::string & err);
    // PB1: a bulk prompt chunk's commit (k: its halves). As committed, with the selector gdn_after_prompt(gdn), but the
    // step keeps 0 rows for a cut: its versions are half ends, so only j = 0 (the chunk's start, the pre-step slot and
    // selector) names a row boundary
    bool committed_prompt(ledger & l, int32_t k, const uint32_t reported[2], const uint32_t reported_gdn[2],
                          std::string & err);
    // ME11: the GDN selector written on both cards with no step recorded (a restore of a saved or checkpointed state,
    // normalized to bank 0 and set 0 with its live steps); refused for a word outside the contract. Synchronous.
    bool set_gdn(ledger & l, uint32_t gdn, std::string & err);

    // FC-RD2: an error word was set on either card, so neither card's commit may take effect: both cards back at the
    // pre-step slot (a card whose own words were clear committed), the ledger unchanged except that nothing stays
    // retained when the failed round ran the head. The caller clears the error words. Synchronous.
    bool restore(ledger & l, std::string & err);

    // The primitive: cut back within the last step to j kept rows and the ledger to n_keep tokens
    // (N_before + j <= n_keep <= L, and n_keep keeps every acknowledged output). Refused, with nothing changed, when no
    // step is recorded, j is outside 0 .. kept, or n_keep is out of range. Synchronous.
    bool cut(ledger & l, int32_t j, int64_t n_keep, std::string & err);

    // after N changed: the summaries' valid blocks become N / 4 (invalidate_from, then the blocks the round pooled)
    static void settle(ledger & l);

    // L = N + U, n_valid = N / 4, 0 <= cur < 6 (SP2: fn::round::n_version), the GDN selector valid (ME11)
    static bool check(ledger & l, std::string & err);

    const step_record & last() const { return last_; }

    // reads both cards' committed-slot words (tests); synchronous
    bool read_cur(uint32_t out[2], std::string & err) const;
    // ME11: and both cards' GDN selectors; synchronous
    bool read_words(uint32_t cur[2], uint32_t gdn[2], std::string & err) const;

  private:
    bool write_cur(uint32_t cur, uint32_t gdn, std::string & err);

    int32_t device_[2] = {-1, -1};
    void * stream_[2] = {nullptr, nullptr};
    uint32_t * cur_dev_[2] = {nullptr, nullptr};
    uint32_t * gdn_dev_[2] = {nullptr, nullptr};
    step_record last_;
};

} // namespace fn::runtime
