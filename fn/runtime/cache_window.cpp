// ME3 (the cleanup): a decode window's GPU work (cache_service.h enqueue), the decode thread's or a capture's; ME1's,
// ME2's and ME2b's code moved here from expert_cache.cpp. The host service's half is cache_service.cpp's.

#include "fn/runtime/cache_service.h"

#include "fn/kernels/experts/experts_cache.h"
#include "fn/xchg/xchg.h"

namespace fn::runtime {

using cache::cuda_ok;
using cache::D;
using cache::fail;

namespace {

// an event record a capture keeps visible to the host and to other streams (cudaEventRecordExternal)
bool record(void * e, void * stream, bool cap, const char * what, std::string & err) {
    return cuda_ok(cap ? cudaEventRecordWithFlags(static_cast<cudaEvent_t>(e), static_cast<cudaStream_t>(stream),
                                                  cudaEventRecordExternal)
                       : cudaEventRecord(static_cast<cudaEvent_t>(e), static_cast<cudaStream_t>(stream)),
                   what, err);
}

} // namespace

bool window_service::enqueue(model::card_ctx * const c[2], const model::layer_weights * const w[2], int32_t rows,
                             fn::experts::gate_up_args ga[2], fn::experts::down_args da[2], int32_t only, int32_t par,
                             bool cap, bool & desc, std::string & err) {
    const cache::devices & dv = *dv_;
    residency & r = *res_;
    desc = false;
    const int32_t k0 = only < 0 ? 0 : only, k1 = only < 0 ? 1 : only;
    const int32_t d0 = only < 0 ? 0 : dv.dev_of(only), d1 = only < 0 ? dv.nd - 1 : dv.dev_of(only);
    const int32_t layer = w[k0]->layer;
    const loader::expert_layer & L = r.layer(layer);
    // card 0's input and routing: the descriptor's copies (device 0)
    const int8_t * xc = nullptr;
    const float * xs = nullptr;
    if (d0 == 0) {
        xc = ga[0].x.codes;
        xs = ga[0].x.scales;
        if (ga[0].hmix != nullptr) {
            xc = reinterpret_cast<const int8_t *>(ga[0].hmix + fn::xchg::hmix_layout::code_at);
            xs = ga[0].hmix + fn::xchg::hmix_layout::scale_at;
        }
        if (xc == nullptr || xs == nullptr || ga[0].ids == nullptr) {
            return fail(err, "a window without its routing or input");
        }
    }
    // per device: the layer's copies (a swap pass); resolve; on device 0 the routing and the FFN input to the
    // layer's staging; the window's event
    for (int32_t d = d0; d <= d1; ++d) {
        const int32_t kd = dv.nd == 1 ? 0 : d; // the card whose stream is the device's
        void * const s = c[kd]->stream;
        if (!dv.set(d, err)) {
            return false;
        }
        if (cap ? !r.replay_wait(d, layer, s, err) : !r.eager_wait(d, layer, s, err)) {
            return false;
        }
        fn::experts::resolve_args ra{};
        ra.ids = ga[kd].ids;
        ra.rows = rows;
        ra.res = reinterpret_cast<const uint8_t * const *>(r.words(d) + r.w_res(layer, 0));
        ra.parts = dv.parts;
        for (int32_t p = 0; p < dv.parts; ++p) {
            ra.off[p] =
                dv.nd == 1 ? ((p & 1) != 0 ? L.dn_at(p / 2) : L.gu_at(p / 2)) : ((p & 1) != 0 ? L.gu_bytes : 0);
        }
        ra.zero = r.zero(d);
        ra.tab = reinterpret_cast<const uint8_t **>(r.words(d) + r.w_tab(0, 0));
        ra.tab2 = reinterpret_cast<const uint8_t **>(r.words(d) + r.w_tab2(0, 0));
        ra.flags = flags_[d] + 2 * layer;
        ra.set = flag_set;
        ra.ext_slots = r.words(d) + r.w_ext();
        ra.run2 = r.words(d) + r.w_run();
        if (ra.ids == nullptr || !fn::experts::resolve(ra, s)) {
            return fail(err, "resolve");
        }
        const cudaStream_t cs = static_cast<cudaStream_t>(s);
        if (d == 0 &&
            (!cuda_ok(cudaMemcpyAsync(ids_h_ + layer * max_b * 10, ga[0].ids, size_t(rows) * 10 * 4,
                                      cudaMemcpyDeviceToHost, cs),
                      "ids", err) ||
             !cuda_ok(cudaMemcpyAsync(x_codes_h_ + int64_t(layer) * max_b * D, xc, size_t(rows) * D,
                                      cudaMemcpyDeviceToHost, cs),
                      "x codes", err) ||
             !cuda_ok(cudaMemcpyAsync(x_scales_h_ + int64_t(layer) * max_b * (D / 32) * 2, xs,
                                      size_t(rows) * (D / 32) * 2 * 4, cudaMemcpyDeviceToHost, cs),
                      "x scales", err))) {
            return false;
        }
        // ME3: a replay's window stamp (a kernel's write, after the stream's earlier work and visible after it), else
        // the window's event
        if (cap) {
            if (!fn::experts::post_word(stamp_d_[d] + layer, 1, s)) {
                return fail(err, "window stamp");
            }
        } else if (!record(ev_desc_[d][size_t(layer)], s, false, "window event", err)) {
            return false;
        }
    }
    desc = true;
    // a flag wait between two timing events (the window's exposed wait)
    const auto wait_flag = [&](int32_t d, int32_t f) {
        const size_t at = size_t(layer * 2 + par) * 4 + size_t(2 * f);
        void * const s = c[dv.nd == 1 ? 0 : d]->stream;
        if (cap) {
            // ME3: a replay's flag wait is a kernel (its times in wt_h_)
            const fn::experts::wait_args wa{flags_[d] + 2 * layer + f, nullptr, flag_set,
                                            wt_d_[d] + wt_flag(layer, par, f)};
            return dv.set(d, err) && (fn::experts::wait_word(wa, s) || fail(err, "flag wait"));
        }
        return dv.set(d, err) && record(ev_fw_[d][at], s, cap, "flag timing", err) &&
               cache::stream_wait(s, flags_[d] + 2 * layer + f, flag_set, "flag wait", err) &&
               record(ev_fw_[d][at + 1], s, cap, "flag timing", err);
    };
    // ME10: one device with both shards in the window: each pass's two gate/up launches as one (gate_up_pair, where
    // supported: cached map 3 at B = 4 and 5), each shard on its own table; and the finish in place of the combine
    const bool one = dv.nd == 1 && only < 0;
    // (a per-card window, only >= 0, may hold a null peer weight pointer: read w[1] only with both cards in it)
    const bool same_fmts = one && w[0]->ex_gu_fmt == w[1]->ex_gu_fmt && w[0]->sh_gate.fmt == w[1]->sh_gate.fmt &&
                           w[0]->sh_up.fmt == w[1]->sh_up.fmt;
    const auto gate_up_pass = [&](const fn::experts::gate_up_args * g, const char * what) {
        if (one && same_fmts) {
            const fn::experts::gate_up_pair_args pa = {{g[0], g[1]}};
            const auto f = static_cast<fn::experts::fmt>(w[0]->ex_gu_fmt);
            const auto fg = static_cast<fn::dense::fmt>(w[0]->sh_gate.fmt);
            const auto fu = static_cast<fn::dense::fmt>(w[0]->sh_up.fmt);
            if (fn::experts::gate_up_pair_supported(f, fg, fu, rows, pa)) {
                return dv.set(0, err) && (fn::experts::gate_up_pair(f, fg, fu, rows, pa, c[0]->stream) ||
                                          fail(err, std::string(what) + " (both cards' pair)"));
            }
        }
        for (int32_t k = k0; k <= k1; ++k) {
            if (!dv.set(dv.dev_of(k), err) ||
                !fn::experts::gate_up(static_cast<fn::experts::fmt>(w[k]->ex_gu_fmt),
                                      static_cast<fn::dense::fmt>(w[k]->sh_gate.fmt),
                                      static_cast<fn::dense::fmt>(w[k]->sh_up.fmt), rows, g[k], c[k]->stream)) {
                return err.empty() ? fail(err, std::string(what) + " (card " + std::to_string(k) + ")") : false;
            }
        }
        return true;
    };
    // pass 1: the resident experts and the shared expert
    for (int32_t k = k0; k <= k1; ++k) {
        const int32_t d = dv.dev_of(k);
        ga[k].w = r.zero(d);
        ga[k].wt = reinterpret_cast<const uint8_t * const *>(r.words(d) + r.w_tab(dv.part_of(k, false), 0));
        ga[k].skip = 1;
    }
    if (!gate_up_pass(ga, "experts gate_up")) {
        return false;
    }
    // pass 2, after the gate/up flag: the misses the service copied to the lane (nothing when the run word is 0)
    for (int32_t d = d0; d <= d1; ++d) {
        if (!wait_flag(d, 0)) {
            return false;
        }
    }
    fn::experts::gate_up_args g2[2] = {ga[0], ga[1]};
    for (int32_t k = k0; k <= k1; ++k) {
        const int32_t d = dv.dev_of(k);
        g2[k].wt = reinterpret_cast<const uint8_t * const *>(r.words(d) + r.w_tab2(dv.part_of(k, false), 0));
        g2[k].skip = 2;
        g2[k].run_if = reinterpret_cast<const uint32_t *>(r.words(d) + r.w_run());
        g2[k].meta = meta2_[k];
    }
    if (!gate_up_pass(g2, "experts gate_up, second pass")) {
        return false;
    }
    for (int32_t k = k0; k <= k1; ++k) {
        if (c[k]->observe != nullptr) {
            c[k]->observe(c[k]->observe_user, k, "ffn.ex_gate_up");
        }
    }
    // ME2b: down at once; ME3: the CPU-served slots (the slot word, written with the tables before the gate/up flag)
    // run no items and take no term in the cache's down instantiation; then on card 0's stream the down flag and the
    // combine, which adds the CPU's products. Only card 0's device waits for the down flag.
    for (int32_t k = k0; k <= k1; ++k) {
        const int32_t d = dv.dev_of(k);
        da[k].w = r.zero(d);
        da[k].wt = reinterpret_cast<const uint8_t * const *>(r.words(d) + r.w_tab(dv.part_of(k, true), 0));
        da[k].ext_slots_d = r.words(d) + r.w_ext();
        if (!dv.set(d, err) ||
            !fn::experts::down(static_cast<fn::experts::fmt>(w[k]->ex_down_fmt),
                               static_cast<fn::experts::fmt>(w[k]->ex_sh_down_fmt), rows, da[k], c[k]->stream)) {
            return err.empty() ? fail(err, "experts down (card " + std::to_string(k) + ")") : false;
        }
    }
    if (k0 == 0) {
        const int32_t dz = dv.dev_of(0);
        fn::experts::combine_args cb{};
        cb.meta = da[0].meta;
        cb.wts = da[0].wts;
        cb.ext_ds = ext_ds_;
        cb.ext_slots = r.words(dz) + r.w_ext();
        cb.out = da[0].out;
        cb.rows = rows;
        if (!wait_flag(dz, 1)) {
            return false;
        }
        if (one) { // ME10: both cards' ws.out[1] written here; layer_pair_step skips X-F's exchange
            float * const half[2] = {c[0]->ws.out[1], c[1]->ws.out[1]};
            if (!fn::experts::combine_finish(cb, da[1].out, half, c[0]->stream)) {
                return fail(err, "the FFN window's finish");
            }
            c[0]->ffn_finished = c[1]->ffn_finished = true;
        } else if (!fn::experts::combine(cb, c[0]->stream)) {
            return fail(err, "the CPU experts' combine");
        }
    }
    for (int32_t k = k0; k <= k1; ++k) {
        if (c[k]->observe != nullptr) {
            c[k]->observe(c[k]->observe_user, k, "ffn.ex_down");
        }
    }
    return true;
}

} // namespace fn::runtime
