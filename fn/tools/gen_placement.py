#!/usr/bin/env python3
"""Generate fn/model/placement.{h,cpp}: the S2 placement of every weight on the two cards (R1).

Inputs (read only, never edited here):
  node-costs.json      fnfl/design/history/wp4: weight_components, each role's source tensors and S2_card_unique_bytes
  memory-account.json  fnfl/design/history/wp2: every tensor's format, shape, stored bytes and S2 card_bytes

The split axis of each tensor follows WP2 design.md (ownership table, sections 2-6): heads, expert channel halves,
vocabulary rows, the router and the shared-expert gate on both cards (LP1: XP1's V2 runs the router on both; DR1: the
draft's too), the indexer on the QSA owner (ordinal % 2; draft: card 0),
HC and PLE feature halves within each of the four streams, GDN value heads 0-7, 16-23, 32-39 on card 0 (Q/K groups
0-7). A card's part of a tensor is a set of rows (ne1 ranges, in every ne2 matrix) times a set of columns (ne0 ranges,
in elements, on quantization-block boundaries).

Checks (the generator fails on any miss): every tensor's rule gives exactly its JSON card_bytes on both cards; every
component's tensors sum exactly to its S2_card_unique_bytes; the card totals equal memory-account's weight_card_bytes;
column ranges sit on block boundaries; stored bytes equal the shape's bytes. The replicated tensors (REPLICATED: every
layer's router and shared-expert gate, the draft's included; LP1, DR1) are the one departure from the JSON: each card
holds the whole tensor, its JSON card_bytes on the owner plus the owner's bytes again on the other card, and the
components and card totals are checked against the JSON plus those copies (each card's added bytes are printed and in
the summary).

usage: gen_placement.py NODE_COSTS MEMORY_ACCOUNT OUT_DIR [SUMMARY_JSON]
"""

import hashlib
import json
import os
import sys

# elements a block, bytes a block (ggml's type traits for the formats the model stores)
FORMATS = {
    'F32': (1, 4), 'F16': (1, 2), 'BF16': (1, 2), 'Q8_0': (32, 34), 'Q2_0': (64, 18), 'IQ4_NL': (32, 18),
    'IQ4_XS': (256, 136), 'IQ3_S': (256, 110), 'IQ3_XXS': (256, 98), 'IQ2_XXS': (256, 66), 'IQ2_XS': (256, 74),
    'IQ2_S': (256, 82), 'Q4_K': (256, 144), 'Q5_K': (256, 176), 'Q6_K': (256, 210),
}
# the EX1 packed layout's formats (experts.h)
EX1_GATE_UP = {'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ3_XXS', 'IQ3_S', 'Q8_0'}
EX1_DOWN = {'Q2_0', 'IQ4_NL', 'Q8_0'}
# the HC layouts' formats (hc.h: BF16, and DR1's Q8_0 planar int8 codes then FP16 scales)
HC_FORMATS = {'BF16', 'Q8_0'}

D = 2560
OWN = 1280
STREAMS = 4
QSA_LAYERS = [l for l in range(48) if l % 4 == 3]


def streams(r):
    """the card's features of each of the four D-wide streams: four ranges of 1,280"""
    return [(s * D + r * OWN, OWN) for s in range(STREAMS)]


def half(n, r):
    return [(r * (n // 2), n // 2)]


def gdn_qkv_rows(r):
    """attn_qkv rows: Q groups 8r..8r+7, K groups 8r..8r+7, V heads 16m+8r..16m+8r+7 (m = 0, 1, 2), 128 each"""
    return [(r * 1024, 1024), (2048 + r * 1024, 1024)] + [(4096 + (16 * m + 8 * r) * 128, 1024) for m in range(3)]


def gdn_v_rows(r, width=128):
    """value heads 16m+8r..16m+8r+7 (m = 0, 1, 2), width values each"""
    return [((16 * m + 8 * r) * width, 8 * width) for m in range(3)]


def full(n):
    return [(0, n)]


def rule(name, ne, r):
    """(rows, cols, layout) for card r; rows/cols are lists of (first, count); None, None: nothing on this card"""
    ne0, ne1 = ne[0], ne[1]
    if name.startswith('blk.'):
        layer = int(name.split('.')[1])
        key = name.split('.', 2)[2]
    else:
        layer, key = None, name
    draft = layer == 48

    def whole_on(owner):
        return (full(ne1), full(ne0), 'gguf') if r == owner else (None, None, 'gguf')

    # global tensors
    if key == 'output.weight':
        return half(ne1, r), full(ne0), 'gguf'  # vocabulary rows
    if key == 'token_embd.weight':
        return full(ne1), half(ne0, r), 'gguf'  # D/2 features of every token
    if key in ('output_hc_down.weight',):
        return full(ne1), streams(r), 'hc_down'
    if key in ('output_hc_norm.weight',):
        return full(ne1), streams(r), 'gguf'
    if key in ('output_hc_up.weight',):
        return streams(r), full(ne0), 'hc_up'
    if key == 'per_layer_token_embd.weight':
        return None, None, 'host'

    # router and shared-expert scalar gate: whole on both cards (LP1; XP1's V2; DR1: the draft's too)
    if key in ('ffn_gate_inp.weight', 'ffn_gate_inp_shexp.weight'):
        return full(ne1), full(ne0), 'gguf'
    # indexer: whole on the QSA owner (ordinal % 2; draft: card 0)
    if key.startswith('indexer.'):
        return whole_on(0 if draft else QSA_LAYERS.index(layer) % 2)
    # replicated small vectors
    if key in ('ssm_norm.weight', 'attn_q_norm.weight', 'attn_k_norm.weight'):
        return full(ne1), full(ne0), 'gguf'

    # routed and shared experts: channels 320r..320r+319 (gate/up rows, down columns)
    if key in ('ffn_gate_exps.weight', 'ffn_up_exps.weight'):
        return half(ne1, r), full(ne0), 'ex1_gate' if key.startswith('ffn_gate') else 'ex1_up'
    if key == 'ffn_down_exps.weight':
        return full(ne1), half(ne0, r), 'ex1_down'
    if key in ('ffn_gate_shexp.weight', 'ffn_up_shexp.weight'):
        return half(ne1, r), full(ne0), 'gguf'
    if key == 'ffn_down_shexp.weight':
        return full(ne1), half(ne0, r), 'gguf'

    # HC: feature halves within each of the four streams
    if key in ('hc_attn_down.weight', 'hc_ffn_down.weight', 'nextn.hc_head_down.weight'):
        return full(ne1), streams(r), 'hc_down'
    if key in ('hc_attn_inject.weight', 'hc_ffn_inject.weight'):
        return full(ne1), streams(r), 'hc_inject'
    if key in ('hc_attn_up.weight', 'hc_ffn_up.weight', 'nextn.hc_head_up.weight'):
        return streams(r), full(ne0), 'hc_up'
    if key in ('hc_attn_norm.weight', 'hc_ffn_norm.weight', 'nextn.hc_head_norm.weight', 'nextn.hnorm.weight'):
        return full(ne1), streams(r), 'gguf'

    # GDN: whole value heads (card 0: 0-7, 16-23, 32-39) and their Q/K groups
    if key == 'attn_qkv.weight':
        return gdn_qkv_rows(r), full(ne0), 'gguf'
    if key == 'attn_gate.weight':
        return gdn_v_rows(r), full(ne0), 'gguf'
    if key in ('ssm_alpha.weight', 'ssm_beta.weight'):
        return gdn_v_rows(r, 1), full(ne0), 'gguf'
    if key in ('ssm_a', 'ssm_dt.bias'):
        return full(ne1), gdn_v_rows(r, 1), 'gguf'
    if key == 'ssm_conv1d.weight':
        return gdn_qkv_rows(r), full(ne0), 'gguf'
    if key == 'ssm_out.weight':
        return full(ne1), gdn_v_rows(r), 'gguf'

    # QSA (target and draft): 12 query heads ([q|gate] interleaved per head) and one KV head
    if key == 'attn_q.weight':
        return half(ne1, r), full(ne0), 'gguf'
    if key in ('attn_k.weight', 'attn_v.weight'):
        return half(ne1, r), full(ne0), 'gguf'
    if key == 'attn_output.weight':
        return full(ne1), half(ne0, r), 'gguf'

    # PLE (layer 1): key and conv over the four streams, value over D, the stream-grouped norms
    if key in ('ple_key.weight', 'ple_conv1d.weight'):
        return streams(r), full(ne0), 'gguf'
    if key == 'ple_value.weight':
        return half(ne1, r), full(ne0), 'gguf'
    if key in ('ple_norm_key.weight', 'ple_norm_query.weight', 'ple_norm_conv.weight'):
        return full(ne1), streams(r), 'gguf'

    # draft input: the card's 1,280 output features of each stream over the whole 5,120-wide input (DR1 Stage C: the
    # cards exchange their halves of the normalized input, then each computes its own outputs)
    if key == 'nextn.eh_proj.weight':
        return half(ne1, r), full(ne0), 'gguf'
    if key == 'nextn.enorm.weight':
        return full(ne1), half(ne0, r), 'gguf'

    raise SystemExit(f'gen_placement: no rule for {name}')


def replicated(name):
    """every layer's router and shared-expert gate, the draft's included, held whole on both cards beyond the S2 account
    (LP1; DR1)"""
    if not name.startswith('blk.'):
        return False
    return name.split('.', 2)[2] in ('ffn_gate_inp.weight', 'ffn_gate_inp_shexp.weight')


def card_bytes(t, rows, cols):
    if rows is None:
        return 0
    blk, tsz = FORMATS[t['format']]
    for first, count in cols:
        if first % blk or count % blk:
            raise SystemExit(f"gen_placement: {t['name']}: column range {first}+{count} is not on {blk}-element blocks")
    ne = t['ne']
    for first, count in rows:
        if first < 0 or first + count > ne[1]:
            raise SystemExit(f"gen_placement: {t['name']}: row range {first}+{count} outside {ne[1]}")
    for first, count in cols:
        if first < 0 or first + count > ne[0]:
            raise SystemExit(f"gen_placement: {t['name']}: column range {first}+{count} outside {ne[0]}")
    n_rows = sum(c for _, c in rows) * ne[2]
    n_cols = sum(c for _, c in cols)
    return n_rows * (n_cols // blk) * tsz


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    nc_path, ma_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    summary_path = sys.argv[4] if len(sys.argv) > 4 else None
    nc = json.load(open(nc_path))
    ma = json.load(open(ma_path))
    comps = nc['weight_components']

    tensors = {}
    for t in ma['tensor_assignments']:
        ne = list(t['shape']) + [1] * (3 - len(t['shape']))
        blk, tsz = FORMATS[t['format']]
        if ne[0] % blk or ne[0] // blk * tsz * ne[1] * ne[2] != t['stored_bytes']:
            raise SystemExit(f"gen_placement: {t['name']}: stored bytes do not match its shape and format")
        tensors[t['name']] = dict(t, ne=ne)

    # component order: node-costs' order; tensors in each component's source order
    comp_list = []
    placed = []
    seen = set()
    extra = [0, 0]  # the replicated tensors' bytes beyond the JSON, per card
    for key, c in comps.items():
        ci = len(comp_list)
        comp_list.append((key, c['role'], -1 if c['layer'] is None else c['layer'], c['S2_card_unique_bytes']))
        sums = [0, 0]
        extra_c = [0, 0]
        for name in c['source_tensors']:
            if name in seen:
                raise SystemExit(f'gen_placement: {name} is in two components')
            seen.add(name)
            t = tensors[name]
            per_card = []
            lay = None
            for r in (0, 1):
                rows, cols, lay_r = rule(name, t['ne'], r)
                if lay_r in ('ex1_gate', 'ex1_up') and t['format'] not in EX1_GATE_UP:
                    lay_r = 'gguf'
                if lay_r == 'ex1_down' and t['format'] not in EX1_DOWN:
                    lay_r = 'gguf'
                if lay_r in ('hc_down', 'hc_inject', 'hc_up') and t['format'] not in HC_FORMATS:
                    lay_r = 'gguf'
                lay = lay_r if lay is None else lay
                if lay != lay_r:
                    raise SystemExit(f'gen_placement: {name}: the cards disagree on the layout')
                b = card_bytes(t, rows, cols)
                want = t['card_bytes'][r]
                if replicated(name):
                    want = max(t['card_bytes'])  # the owner's whole tensor on each card
                    if sorted(t['card_bytes']) != [0, want]:
                        raise SystemExit(f"gen_placement: {name}: memory-account.json does not hold it on one card")
                    extra_c[r] += want - t['card_bytes'][r]
                if b != want:
                    raise SystemExit(f"gen_placement: {name} card {r}: the rule gives {b} bytes, expected {want} "
                                     f"(memory-account.json {t['card_bytes'][r]})")
                sums[r] += b
                per_card.append((rows, cols, b))
            placed.append(dict(name=name, comp=ci, t=t, lay=lay, cards=per_card))
        want_c = [c['S2_card_unique_bytes'][r] + extra_c[r] for r in (0, 1)]
        if sums != want_c:
            raise SystemExit(f"gen_placement: component {key}: tensors sum to {sums}, S2_card_unique_bytes "
                             f"{c['S2_card_unique_bytes']} + replicated {extra_c}")
        for r in (0, 1):
            extra[r] += extra_c[r]
        comp_list[ci] = comp_list[ci][:3] + (want_c,)
    if seen != set(tensors):
        raise SystemExit(f'gen_placement: tensors outside every component: {sorted(set(tensors) - seen)}')

    totals = [sum(p['cards'][r][2] for p in placed) for r in (0, 1)]
    want_t = [ma['weight_card_bytes'][r] + extra[r] for r in (0, 1)]
    if totals != want_t:
        raise SystemExit(f"gen_placement: card totals {totals} != weight_card_bytes {ma['weight_card_bytes']} + "
                         f"replicated {extra}")

    # regions: tensors that share one device buffer (HC down + inject; EX1 gate + up), else one each
    region_of = {}
    regions = []
    for i, p in enumerate(placed):
        name, lay = p['name'], p['lay']
        if lay in ('hc_inject', 'ex1_up', 'host'):
            continue
        region_of[i] = len(regions)
        regions.append(lay)
    for i, p in enumerate(placed):
        if p['lay'] == 'hc_inject':
            mate = p['name'].replace('_inject', '_down')
        elif p['lay'] == 'ex1_up':
            mate = p['name'].replace('ffn_up_exps', 'ffn_gate_exps')
        else:
            continue
        j = next(k for k, q in enumerate(placed) if q['name'] == mate)
        if placed[j]['lay'] not in ('hc_down', 'ex1_gate'):
            raise SystemExit(f"gen_placement: {p['name']}: its region partner {mate} has layout {placed[j]['lay']}")
        region_of[i] = region_of[j]

    ranges = []

    def add_ranges(rs):
        if rs is None:
            return -1, 0
        first = len(ranges)
        ranges.extend(rs)
        return first, len(rs)

    rows_out = []
    for i, p in enumerate(placed):
        cards = []
        for r in (0, 1):
            rows, cols, b = p['cards'][r]
            rf, rc = add_ranges(rows)
            cf, cc = add_ranges(cols)
            cards.append((rf, rc, cf, cc, b))
        rows_out.append((p, region_of.get(i, -1), cards))

    src_hash = {'node_costs': sha256_file(nc_path), 'memory_account': sha256_file(ma_path)}
    gen_hash = sha256_file(os.path.abspath(__file__))

    h = []
    h.append('#pragma once')
    h.append('')
    h.append('// GENERATED by fn/tools/gen_placement.py (R1); do not edit. The S2 placement of every weight on the two cards,')
    h.append('// from fnfl/design/history/wp4/node-costs.json (weight_components) and wp2/memory-account.json')
    h.append('// (tensor_assignments):')
    h.append(f"//   node-costs.json      sha256 {src_hash['node_costs']}")
    h.append(f"//   memory-account.json  sha256 {src_hash['memory_account']}")
    h.append(f'//   gen_placement.py     sha256 {gen_hash}')
    h.append('// A card holds, of a tensor [ne0, ne1, ne2] (ne0 contiguous), the rows in its row ranges (ne1 indices, in every')
    h.append('// ne2 matrix) times the columns in its column ranges (ne0 elements, on block boundaries). Its slice order is')
    h.append('// matrix, then row range and row, then column range: the GGUF bytes of the slice read in file order. The layout')
    h.append('// says how the slice is laid out on the card (fn/loader/loader.h). The generator checked every byte count against')
    h.append('// the JSON: per tensor, per component (S2_card_unique_bytes) and per card (weight_card_bytes), except that the')
    h.append("// router and shared-expert gate of every layer (the draft's too) are whole on both cards (LP1, DR1): their")
    h.append(f'// copies on the non-owner card add {extra[0]} / {extra[1]} bytes to cards 0 / 1, counted in the components and')
    h.append('// the totals below.')
    h.append('')
    h.append('#include <cstdint>')
    h.append('')
    h.append('namespace fn::model {')
    h.append('')
    h.append('enum class layout : int32_t {')
    h.append('    gguf,      // the slice in slice order')
    h.append('    hc_down,   // fn::hc w_down [s][j][i] (j < 320; rows 320..323 hold the region\'s inject tensor, if any);')
    h.append('               // Q8_0: the int8 codes [s][j][i], then the FP16 scales [s][j][i / 32] (hc.h)')
    h.append('    hc_inject, // fn::hc w_down rows 320..323, in its down tensor\'s region')
    h.append('    hc_up,     // fn::hc w_up [c][s][j]; Q8_0: the codes [c][s][j], then the scales [c][s][j / 32]')
    h.append('    ex1_gate,  // fn::experts packed gate/up [512][gate_up_bytes], with its up tensor (pack_gate_up)')
    h.append('    ex1_up,    // in its gate tensor\'s region')
    h.append('    ex1_down,  // fn::experts packed down [512][down_bytes] (pack_down)')
    h.append('    host,      // stays mapped on the host (the PLE table)')
    h.append('};')
    h.append('')
    h.append('struct range {')
    h.append('    int32_t first;')
    h.append('    int32_t count;')
    h.append('};')
    h.append('')
    h.append('struct card_place {')
    h.append('    int32_t row_first, row_count; // ranges[row_first ..], ne1 indices; row_count 0: nothing on this card')
    h.append('    int32_t col_first, col_count; // ranges[col_first ..], ne0 elements')
    h.append('    int64_t bytes;                // memory-account.json card_bytes (a replicated tensor: the whole tensor)')
    h.append('};')
    h.append('')
    h.append('struct tensor_place {')
    h.append('    const char * name;')
    h.append('    int32_t component;  // components[]')
    h.append('    const char * format; // ggml type name')
    h.append('    int32_t block_elems; // elements a quantization block')
    h.append('    int32_t block_bytes;')
    h.append('    int64_t ne[3];')
    h.append('    int64_t stored_bytes;')
    h.append('    layout lay;')
    h.append('    int32_t region; // tensors with the same region share one buffer on each card; -1 for host')
    h.append('    card_place card[2];')
    h.append('};')
    h.append('')
    h.append('struct component {')
    h.append('    const char * key;  // node-costs.json weight_components key')
    h.append('    const char * role;')
    h.append('    int32_t layer;     // -1: global; 48: the draft block')
    h.append('    int64_t card_bytes[2]; // S2_card_unique_bytes, plus the replicated tensors\' copies')
    h.append('};')
    h.append('')
    h.append(f'constexpr int32_t n_tensors = {len(rows_out)};')
    h.append(f'constexpr int32_t n_components = {len(comp_list)};')
    h.append(f'constexpr int32_t n_regions = {len(regions)};')
    h.append(f'constexpr int32_t n_ranges = {len(ranges)};')
    h.append(f'constexpr int64_t weight_card_bytes[2] = {{{totals[0]}, {totals[1]}}}; // memory-account.json + replicated')
    h.append('')
    h.append('extern const tensor_place tensors[n_tensors];')
    h.append('extern const component components[n_components];')
    h.append('extern const range ranges[n_ranges];')
    h.append('')
    h.append('} // namespace fn::model')
    h.append('')

    c = []
    c.append('// GENERATED by fn/tools/gen_placement.py (R1); do not edit. See placement.h.')
    c.append('')
    c.append('#include "fn/model/placement.h"')
    c.append('')
    c.append('namespace fn::model {')
    c.append('')
    c.append('const tensor_place tensors[n_tensors] = {')
    for p, reg, cards in rows_out:
        t = p['t']
        blk, tsz = FORMATS[t['format']]
        cs = ', '.join(f'{{{a}, {b}, {cf}, {cc}, {by}}}' for a, b, cf, cc, by in cards)
        c.append(f'    {{"{p["name"]}", {p["comp"]}, "{t["format"]}", {blk}, {tsz}, {{{t["ne"][0]}, {t["ne"][1]}, '
                 f'{t["ne"][2]}}}, {t["stored_bytes"]},')
        c.append(f'     layout::{p["lay"]}, {reg}, {{{cs}}}}},')
    c.append('};')
    c.append('')
    c.append('const component components[n_components] = {')
    for key, role, layer, cb in comp_list:
        c.append(f'    {{"{key}", "{role}", {layer}, {{{cb[0]}, {cb[1]}}}}},')
    c.append('};')
    c.append('')
    c.append('const range ranges[n_ranges] = {')
    line = '   '
    for a, b in ranges:
        item = f' {{{a}, {b}}},'
        if len(line) + len(item) > 116:
            c.append(line)
            line = '   '
        line += item
    if line.strip():
        c.append(line)
    c.append('};')
    c.append('')
    c.append('} // namespace fn::model')
    c.append('')

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'placement.h'), 'w') as f:
        f.write('\n'.join(h))
    with open(os.path.join(out_dir, 'placement.cpp'), 'w') as f:
        f.write('\n'.join(c))

    lays = {}
    for p in placed:
        for r in (0, 1):
            lays.setdefault(p['lay'], [0, 0])[r] += p['cards'][r][2]
    roles = {}
    for p in placed:
        role = comp_list[p['comp']][1]
        for r in (0, 1):
            roles.setdefault(role, [0, 0])[r] += p['cards'][r][2]
    summary = dict(sources=src_hash, generator_sha256=gen_hash, n_tensors=len(placed), n_components=len(comp_list),
                   n_regions=len(regions), card_bytes=totals,
                   card_GiB=[x / 2**30 for x in totals], replicated_bytes=extra, bytes_by_layout=lays, bytes_by_role=roles,
                   checks=['tensor card_bytes exact (replicated: the whole tensor on each card)',
                           'component S2_card_unique_bytes + replicated copies exact',
                           'card totals == weight_card_bytes + replicated copies', 'column ranges on block boundaries',
                           'stored bytes == shape bytes'])
    if summary_path:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=1)
    print(f'gen_placement: PASS: {len(placed)} tensors, {len(comp_list)} components, {len(regions)} regions; '
          f'card bytes {totals[0]} / {totals[1]} ({totals[0] / 2**30:.6f} / {totals[1] / 2**30:.6f} GiB), every '
          f'tensor and component exact; replicated routers add {extra[0]} / {extra[1]} bytes')


if __name__ == '__main__':
    main()
