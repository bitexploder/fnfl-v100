#!/usr/bin/env python3
"""The integrated build's captured graphs against a named baseline's: every declared component change, and nothing else
(Shell's integration preflight, "Combined build: compare the declared graph changes together"; Fable's plan f598f442).

usage: integrated_nodes.py --baseline NAME [--components LIST] [--two-device] BASE_DIR INTEGRATED_DIR
       integrated_nodes.py --self-test     (CPU fixtures; no GPU)

The lists are model-dry's OUT/graph-nodes-*.txt (target::write_graph_nodes: "I kernel NAME grid X,Y,Z block X,Y,Z" or
"I type T [label]"). BASELINE names the build BASE_DIR came from, which sets the changes between it and the integrated
build:
  main-d7f8a885  retained main (ME7 in, ME8 not): ME8, ME10, ME12 and ME11 apply;
  q2-ccc293f     the bundle's Q2 part (ME5/ME9/ME8 in, tree = e0b112d): ME10, ME12 and ME11 apply, not ME8 again.
--components narrows that set (for a component-only pair of listings, e.g. me11 on request 16's own lists).

One device (the default), exactly, in counts of normalized records, per graph:
  ME8,  target pass graph B: dense product_kernel<B,false> -48 (card 1's router a layer);
        draft step graph of d proposals: dense product_kernel<1,true> -d (card 1's router a proposal); catch-up: none.
  ME10, target pass graph B: xchg send_half -96 and recv_add -96 (X-F's), experts combine_kernel -48, experts
        finish_kernel +48 (grid 10, block 256); at B 4 and 5 every gate_up_kernel_u<X,FAM,B,D> instance -2n and the
        gate_up_kernel_u_pair of the same template arguments +n (grid 72, block 640), 192 removed and 96 added in all;
        draft and catch-up: none.
  ME12, target pass graph B: hc_sub_kernel<B,false> -384 and hc_sub_kernel<B,true> -4 (the head's), hc_pair_kernel
        <B,false> +96 and hc_pair_kernel<B,true> +1 (grid 72, block 640); draft and catch-up (Q8_0 HC): none.
  ME11, every graph: no node; kernel names compared after the one normalization below.
--two-device: every graph identical (ME8, ME10 and ME12 keep the two-device paths; ME11 adds no node).

Kernel identity is the whole record: the symbol (family, template arguments, argument types) and the launch shape.
Only the eight hex digits after "_GLOBAL__N__" (a translation unit's per-compilation component) are normalized; two
records of one list that normalize alike fail as a collapse. Order: the integrated list with the declared additions
dropped must be the baseline's list with exactly the declared removals deleted, in order (a subsequence with the
right multiset of deletions). The file sets must match, with target pass graphs B1..B5 and draft and catch-up graphs.
No family is ignored: a change outside these rules, a count off by one, a wrong B or launch shape, or a reordering
fails. Exit 0 for PASS.
"""
import collections
import os
import re
import shutil
import sys
import tempfile

GLOBAL_HASH = re.compile(r'(_GLOBAL__N__)[0-9a-f]{8}(?=_)')
NAME = re.compile(r'^graph-nodes-(?:(draft|catchup)-(?:sparse|dense)-)?(?:v\d+-)?B(\d+)-card(\d+)\.txt$')
BASELINES = {'main-d7f8a885': ('me8', 'me10', 'me12', 'me11'), 'q2-ccc293f': ('me10', 'me12', 'me11')}

# identities, as patterns on a normalized record ("kernel SYMBOL grid X,Y,Z block X,Y,Z")
DENSE = r'^kernel _ZN2fn5dense\d+_GLOBAL__N__00000000_\d+_dense_cu_[0-9a-f]{8}\d+product_kernelILi%dELb%dEEEvNS1_11launch_descE grid 72,1,1 block 512,1,1$'
SEND_HALF = r'^kernel _ZN2fn4xchg\d+_GLOBAL__N__00000000_\d+_layer_xchg_cu_[0-9a-f]{8}\d+send_half_kernelEPKfiPji grid \d+,1,1 block 128,1,1$'
RECV_ADD = r'^kernel _ZN2fn4xchg\d+_GLOBAL__N__00000000_\d+_layer_xchg_cu_[0-9a-f]{8}\d+recv_add_kernelEPKfiPjPfS4_ji grid \d+,1,1 block 128,1,1$'
COMBINE = r'^kernel _ZN2fn7experts\d+_GLOBAL__N__00000000_\d+_experts_cache_cu_[0-9a-f]{8}\d+combine_kernelENS0_12combine_argsE grid 10,1,1 block 256,1,1$'
FINISH = r'^kernel _ZN2fn7experts\d+_GLOBAL__N__00000000_\d+_experts_cache_cu_[0-9a-f]{8}\d+finish_kernelENS0_12combine_argsEPKfPfS5_ grid 10,1,1 block 256,1,1$'
GU = re.compile(r'^kernel (_ZN2fn7experts\d+_GLOBAL__N__00000000_\d+_experts_gu_\w+?_cu_[0-9a-f]{8})16gate_up_kernel_uI(Lb[01]ENS1_\d+gu_\w+?ELi(\d)ELi\d+E)EEvNS0_12gate_up_argsE grid 72,1,1 block 640,1,1$')
GU_PAIR = re.compile(r'^kernel (_ZN2fn7experts\d+_GLOBAL__N__00000000_\d+_experts_gu_\w+?_cu_[0-9a-f]{8})21gate_up_kernel_u_pairI(Lb[01]ENS1_\d+gu_\w+?ELi(\d)ELi\d+E)EEvNS0_17gate_up_pair_argsE grid (\S+) block (\S+)$')
HC_SUB = r'^kernel _ZN2fn2hc13hc_sub_kernelILi%dELb%dEEEvNS0_10sub_paramsEi grid 72,1,1 block 640,1,1$'
HC_PAIR = r'^kernel _ZN2fn2hc14hc_pair_kernelILi%dELb%dEEEvNS0_11pair_paramsE grid 72,1,1 block 640,1,1$'


def records(path):
    seq, raw = [], collections.defaultdict(set)
    with open(path, errors='replace') as f:
        for line in f:
            parts = line.split(None, 1)
            if len(parts) != 2 or not parts[0].isdigit():
                raise ValueError('%s: an unparsed line: %r' % (path, line))
            r = parts[1].strip()
            n = GLOBAL_HASH.sub(r'\g<1>00000000', r)
            raw[n].add(r)
            seq.append(n)
    return seq, sorted(n for n, rs in raw.items() if len(rs) > 1)


def matching(pattern, counter):
    rx = re.compile(pattern)
    return [r for r in counter if rx.search(r)]


def expected(kind, rows, comps, base):
    """the declared (removals, additions) as Counters of exact records, built from the baseline's own records, or an
    error string"""
    rem, add = collections.Counter(), collections.Counter()
    errs = []

    def take(pattern, n, what):
        ids = matching(pattern, base)
        if len(ids) != 1:
            errs.append('%s: %d baseline records match, expected exactly one identity' % (what, len(ids)))
            return None
        if base[ids[0]] < n:
            errs.append('%s: the baseline holds %d, fewer than the %d to remove' % (what, base[ids[0]], n))
        rem[ids[0]] += n
        return ids[0]

    def put(record, n):
        add[record] += n

    if kind == 'target':
        if 'me8' in comps:
            take(DENSE % (rows, 0), 48, 'ME8 router product_kernel<%d,false>' % rows)
        if 'me10' in comps:
            take(SEND_HALF, 96, 'ME10 send_half')
            take(RECV_ADD, 96, 'ME10 recv_add')
            cb = take(COMBINE, 48, 'ME10 combine')
            if cb is not None:
                put(re.sub(r'14combine_kernelENS0_12combine_argsE', '13finish_kernelENS0_12combine_argsEPKfPfS5_', cb),
                    48)
            if rows >= 4:
                n_gu = 0
                for r, c in base.items():
                    m = GU.match(r)
                    if m and int(m.group(3)) == rows:
                        if c % 2:
                            errs.append('ME10: an odd count %d of %s' % (c, r))
                        rem[r] += c
                        n_gu += c
                        put('kernel %s21gate_up_kernel_u_pairI%sEEvNS0_17gate_up_pair_argsE grid 72,1,1 block 640,1,1' %
                            (m.group(1), m.group(2)), c // 2)
                if n_gu != 192:
                    errs.append('ME10: the baseline holds %d gate_up_kernel_u<B %d> records, expected 192' % (n_gu, rows))
        if 'me12' in comps:
            take(HC_SUB % (rows, 0), 384, 'ME12 hc_sub<%d,false>' % rows)
            take(HC_SUB % (rows, 1), 4, 'ME12 hc_sub<%d,true>' % rows)
            put('kernel _ZN2fn2hc14hc_pair_kernelILi%dELb0EEEvNS0_11pair_paramsE grid 72,1,1 block 640,1,1' % rows, 96)
            put('kernel _ZN2fn2hc14hc_pair_kernelILi%dELb1EEEvNS0_11pair_paramsE grid 72,1,1 block 640,1,1' % rows, 1)
    elif kind == 'draft' and 'me8' in comps:
        take(DENSE % (1, 1), rows, 'ME8 draft router product_kernel<1,true>')
    return rem, add, errs


def subsequence_deletes(base_seq, cand_seq):
    """cand_seq as an in-order subsequence of base_seq: the Counter of deleted base records, or None"""
    deleted = collections.Counter()
    i = 0
    for r in cand_seq:
        while i < len(base_seq) and base_seq[i] != r:
            deleted[base_seq[i]] += 1
            i += 1
        if i == len(base_seq):
            return None
        i += 1
    for r in base_seq[i:]:
        deleted[r] += 1
    return deleted


def check(baseline, comps, two_device, before, after, out=print):
    fails = []
    fa = sorted(f for f in os.listdir(before) if f.startswith('graph-nodes'))
    fb = sorted(f for f in os.listdir(after) if f.startswith('graph-nodes'))
    if fa != fb:
        fails.append('file sets differ: only baseline %s, only integrated %s' %
                     (sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))))
    kinds = collections.defaultdict(set)
    for f in fa:
        m = NAME.match(f)
        if m is None:
            fails.append('%s: not a known graph listing name' % f)
        else:
            kinds[m.group(1) or 'target'].add(int(m.group(2)))
    for k in ('target', 'draft', 'catchup'):
        if not kinds[k]:
            fails.append('no %s graph listed' % k)
    if kinds['target'] and sorted(kinds['target']) != list(range(1, 6)):
        fails.append('target pass graphs not B1..B5: %s' % sorted(kinds['target']))
    total = [0, 0]
    for f in sorted(set(fa) & set(fb)):
        m = NAME.match(f)
        if m is None:
            continue
        kind, rows = m.group(1) or 'target', int(m.group(2))
        (sx, cx), (sy, cy) = records(os.path.join(before, f)), records(os.path.join(after, f))
        total[0] += len(sx)
        total[1] += len(sy)
        bad = ['%s: distinct records collapse under the name normalization: %s' % (side, col)
               for side, col in (('baseline', cx), ('integrated', cy)) if col]
        x = collections.Counter(sx)
        rem, add, errs = (collections.Counter(), collections.Counter(), []) if two_device else \
            expected(kind, rows, comps, x)
        bad += errs
        added_ids = set(add)
        if added_ids & set(x):
            bad.append('a declared addition already in the baseline: %s' % sorted(added_ids & set(x)))
        y = collections.Counter(sy)
        got_add = collections.Counter({r: y[r] for r in added_ids})
        if got_add != add:
            bad.append('declared additions %s, found %s' % (dict(add), dict(got_add)))
        kept = [r for r in sy if r not in added_ids]
        deleted = subsequence_deletes(sx, kept)
        if deleted is None:
            bad.append('the integrated records (declared additions dropped) are not the baseline\'s in order: an '
                       'extra, changed or reordered record')
        else:
            extra = deleted - rem
            missing = rem - deleted
            if extra:
                bad.append('removed but not declared: %s' % dict(extra))
            if missing:
                bad.append('declared removals not removed: %s' % dict(missing))
        out('%-52s %s' % (f, ('ok: %d -> %d nodes, -%d +%d as declared' %
                               (len(sx), len(sy), sum(rem.values()), sum(add.values()))) if not bad
                          else 'FAIL: ' + '; '.join(bad)))
        fails.extend('%s: %s' % (f, b) for b in bad)
    for x in fails:
        out('integrated_nodes: FAIL: %s' % x)
    out('integrated_nodes: baseline %s, components %s%s: %d graphs, %d -> %d nodes: %s (%d failures)' %
        (baseline, ','.join(comps) if not two_device else 'none (two devices)', ' (two devices)' if two_device else '',
         len(fa), total[0], total[1], 'PASS' if not fails else 'FAIL', len(fails)))
    return not fails


# ---- self-test: synthetic listings built from the real identities, the integrated form derived by the rules ----

def _sym():
    h = '4d5af096'
    return {
        'dense': lambda b, t: 'kernel _ZN2fn5dense36_GLOBAL__N__%s_8_dense_cu_1a2b3c4d14product_kernelILi%dELb%dEEEvNS1_11launch_descE grid 72,1,1 block 512,1,1' % (h, b, t),
        'sh': 'kernel _ZN2fn4xchg36_GLOBAL__N__%s_13_layer_xchg_cu_5e6f7a8b16send_half_kernelEPKfiPji grid 13,1,1 block 128,1,1' % h,
        'ra': 'kernel _ZN2fn4xchg36_GLOBAL__N__%s_13_layer_xchg_cu_5e6f7a8b15recv_add_kernelEPKfiPjPfS4_ji grid 13,1,1 block 128,1,1' % h,
        'cb': 'kernel _ZN2fn7experts49_GLOBAL__N__%s_16_experts_cache_cu_847012d114combine_kernelENS0_12combine_argsE grid 10,1,1 block 256,1,1' % h,
        'fi': 'kernel _ZN2fn7experts49_GLOBAL__N__%s_16_experts_cache_cu_847012d113finish_kernelENS0_12combine_argsEPKfPfS5_ grid 10,1,1 block 256,1,1' % h,
        'gu': lambda fam, b: 'kernel _ZN2fn7experts50_GLOBAL__N__%s_17_experts_gu_%s_cu_1d36d60916gate_up_kernel_uILb1ENS1_6gu_%sELi%dELi2EEEvNS0_12gate_up_argsE grid 72,1,1 block 640,1,1' % (h, fam, fam, b),
        'gp': lambda fam, b, g='72,1,1': 'kernel _ZN2fn7experts50_GLOBAL__N__%s_17_experts_gu_%s_cu_1d36d60921gate_up_kernel_u_pairILb1ENS1_6gu_%sELi%dELi2EEEvNS0_17gate_up_pair_argsE grid %s block 640,1,1' % (h, fam, fam, b, g),
        'hs': lambda b, hd: 'kernel _ZN2fn2hc13hc_sub_kernelILi%dELb%dEEEvNS0_10sub_paramsEi grid 72,1,1 block 640,1,1' % (b, hd),
        'hp': lambda b, hd, g='72,1,1': 'kernel _ZN2fn2hc14hc_pair_kernelILi%dELb%dEEEvNS0_11pair_paramsE grid %s block 640,1,1' % (b, hd, g),
        'gdn': lambda b: 'kernel _ZN2fn3gdn38_GLOBAL__N__%s_6_gdn_cu_5786f24a11core_kernelILi%dEEEvNS0_9core_argsE grid 144,1,1 block 704,1,1' % (h, b),
        'mem': 'type 1 memcpy',
    }


def _target(b, comps, sab=''):
    s = _sym()
    lines = []
    for layer in range(48):
        for sub in range(2):
            if 'me12' in comps:
                lines += [s['hp'](b, 0, '36,1,1' if sab == 'hc-shape' else '72,1,1')] if not (sab == 'hc-missing' and layer == 0) else []
            else:
                lines += [s['hs'](b, 0)] * 4
            if sub == 0:
                lines += [s['gdn'](b if sab != 'gdn-b' or layer else 1 + b % 5), s['dense'](b, 0), s['dense'](b, 0)]
                lines += [s['sh'], s['sh'], s['ra'], s['ra']]  # X-A
            else:
                lines += [s['dense'](b, 0)] * (1 if 'me8' in comps else 2)  # routers
                fam = 'iq2' if layer % 2 else 'iq3'
                if b >= 4 and 'me10' in comps:
                    lines += [s['gp'](fam, b if sab != 'pair-b' else 9 - b, '36,1,1' if sab == 'pair-shape' else '72,1,1')] * 2
                else:
                    lines += [s['gu'](fam, b)] * 4
                lines += [s['mem']]
                if 'me10' in comps:
                    lines += [s['fi']]
                else:
                    lines += [s['cb'], s['sh'], s['sh'], s['ra'], s['ra']]  # X-F
    lines += ([s['hp'](b, 1)] if 'me12' in comps else [s['hs'](b, 1)] * 4) + [s['mem']]
    if sab == 'extra':
        lines.insert(10, s['dense'](b, 1))
    if sab == 'swap':
        lines[3], lines[4] = lines[4], lines[3]
    if sab == 'drop-mem':
        lines.remove(s['mem'])
    return lines


def _draft(d, comps):
    s = _sym()
    lines = []
    for _ in range(d):
        lines += [s['dense'](1, 0), s['dense'](1, 1)] + ([] if 'me8' in comps else [s['dense'](1, 1)]) + [s['mem']]
    return lines


def _write(root, comps, sab='', drop=None, h2=None):
    os.makedirs(root)
    files = {}
    for b in range(1, 6):
        files['graph-nodes-B%d-card0.txt' % b] = _target(b, comps, sab)
    for d in range(1, 5):
        files['graph-nodes-draft-sparse-v0-B%d-card0.txt' % d] = _draft(d, comps)
        files['graph-nodes-catchup-sparse-v0-B%d-card0.txt' % d] = [_sym()['dense'](d, 0), _sym()['mem']]
    for name, lines in files.items():
        if name == drop:
            continue
        text = '\n'.join('%d %s' % (i, l) for i, l in enumerate(lines)) + '\n'
        if h2 is not None:
            text = text.replace('4d5af096', h2)
        open(os.path.join(root, name), 'w').write(text)


def self_test():
    tmp = tempfile.mkdtemp()
    ok = True
    try:
        all4 = ('me8', 'me10', 'me12', 'me11')
        _write(os.path.join(tmp, 'main'), ())
        _write(os.path.join(tmp, 'q2'), ('me8',))
        # (what, baseline dir, components the check applies, components the integrated fixture was built with,
        # sabotage, a dropped file, another compilation component, expected verdict)
        cases = [
            ('main baseline: every declared change', 'main', all4, all4, '', None, None, True),
            ('main baseline: another compilation component', 'main', all4, all4, '', None, '0bad7311', True),
            ('q2 baseline (ME8 in): ME10, ME12, ME11', 'q2', ('me10', 'me12', 'me11'), all4, '', None, None, True),
            ('q2 baseline but ME8 subtracted again', 'q2', all4, all4, '', None, None, False),
            ('main baseline without ME8 counted', 'main', ('me10', 'me12', 'me11'), all4, '', None, None, False),
            ('ME12 not applied (HC sub nodes kept)', 'main', all4, ('me8', 'me10', 'me11'), '', None, None, False),
            ('a missing HC pair node', 'main', all4, all4, 'hc-missing', None, None, False),
            ('an extra node', 'main', all4, all4, 'extra', None, None, False),
            ('a dropped unrelated memcpy', 'main', all4, all4, 'drop-mem', None, None, False),
            ('the HC pair at 36 CTAs', 'main', all4, all4, 'hc-shape', None, None, False),
            ('the gate/up pair at 36 CTAs', 'main', all4, all4, 'pair-shape', None, None, False),
            ('the gate/up pair of the wrong B', 'main', all4, all4, 'pair-b', None, None, False),
            ('an unrelated GDN kernel of the wrong B', 'main', all4, all4, 'gdn-b', None, None, False),
            ('two records reordered', 'main', all4, all4, 'swap', None, None, False),
            ('a missing B4 graph', 'main', all4, all4, '', 'graph-nodes-B4-card0.txt', None, False),
        ]
        for i, (what, base, comps, built, sab, drop, h2, want) in enumerate(cases):
            root = os.path.join(tmp, 'after%d' % i)
            _write(root, built, sab, drop, h2)
            got = check('main-d7f8a885' if base == 'main' else 'q2-ccc293f', comps, False,
                        os.path.join(tmp, base), root, out=lambda s: None)
            print('integrated_nodes self-test: %-46s %s (%s)' % (what, 'ok' if got == want else 'WRONG',
                                                                  'passes' if got else 'fails'))
            ok = ok and got == want
        # two devices: identical passes, any change fails
        _write(os.path.join(tmp, 'two-a'), ())
        _write(os.path.join(tmp, 'two-b'), ())
        _write(os.path.join(tmp, 'two-c'), (), sab='extra')
        for what, root, want in (('two devices: identical', 'two-b', True), ('two devices: an extra node', 'two-c', False)):
            got = check('main-d7f8a885', (), True, os.path.join(tmp, 'two-a'), os.path.join(tmp, root), out=lambda s: None)
            print('integrated_nodes self-test: %-46s %s (%s)' % (what, 'ok' if got == want else 'WRONG',
                                                                  'passes' if got else 'fails'))
            ok = ok and got == want
        print('integrated_nodes self-test: %s' % ('PASS' if ok else 'FAIL'))
        return ok
    finally:
        shutil.rmtree(tmp)


def main(argv):
    if argv == ['--self-test']:
        return 0 if self_test() else 1
    args = list(argv)
    baseline, comps, two = None, None, False
    while args and args[0].startswith('--'):
        a = args.pop(0)
        if a == '--baseline' and args:
            baseline = args.pop(0)
        elif a == '--components' and args:
            comps = tuple(c for c in args.pop(0).split(',') if c)
        elif a == '--two-device':
            two = True
        else:
            print(__doc__)
            return 2
    if baseline not in BASELINES or len(args) != 2:
        print(__doc__)
        return 2
    allowed = BASELINES[baseline]
    comps = allowed if comps is None else comps
    if any(c not in allowed for c in comps):
        print('integrated_nodes: components %s are not all between baseline %s and the integrated build (%s)' %
              (','.join(comps), baseline, ','.join(allowed)))
        return 2
    return 0 if check(baseline, comps, two, args[0], args[1]) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
