#!/usr/bin/env python3
"""ME8: assert the routers ME8 removes from the one-device captured graphs, and nothing else (Shell's review,
6ff53d10). Reads two builds' node lists (model-dry's OUT/graph-nodes-*.txt, target::write_graph_nodes: one node a
line, "I kernel NAME grid X,Y,Z block X,Y,Z" or "I type T [label]"), BEFORE (ME7's build) and AFTER (ME8's).

Asserts, failing (exit 1) on any miss:
  - the same, complete file sets: every file in one is in the other; at least one target pass graph (B<n>), one draft
    step graph and one catch-up graph; the target pass graphs B1..B<max> with no gap;
  - each file's node records compared as multisets, the enumeration index alone discarded: AFTER adds nothing, and
    removes exactly the graph's routers: a target pass graph 48 (one a layer), a draft step graph B<rows> (one a
    proposal), a catch-up graph none (no FFN), from its file name's capture kind and rows;
  - every removed record is one kernel identity (name and launch shape), and AFTER still holds at least as many of
    that identity (card 0's routers, which remain);
  - no unreadable node ("type -1 unread") or unnamed kernel ("kernel ?") in either list differs.

Names: a kernel in an anonymous namespace carries its translation unit's per-compilation component, eight hex digits
after "_GLOBAL__N__" (e.g. _GLOBAL__N__4d5af096_8_round_cu_973253cc), which differs between two builds of the same
source (Shell, request 10). Only those eight digits are normalized (to 00000000); the file component, the rest of the
symbol, the launch shape and every non-kernel record are compared as written. Two distinct records of one list that
normalize to the same record are an ambiguous collapse and fail.

ME10 (--me10 BEFORE AFTER: BEFORE the ME8 descendant's one-device cached graphs, AFTER ME10's): each target pass
graph replaces its 48 layers' X-F exchange and combine by the finish, and at B 4 and 5 each layer's four gate/up
launches by two pair launches. Exactly, in multiset counts of the normalized records:
  every target pass graph: send_half_kernel -96 and recv_add_kernel -96 (X-F's two a layer; X-A's stay),
    combine_kernel -48, finish_kernel +48 (grid 10, block 256);
  B 4 and 5 also: gate_up_kernel_u (its cache instances) -192, gate_up_kernel_u_pair +96 (grid 72, block 640);
  draft and catch-up graphs: nothing.
Nothing else is added or removed, the records outside those identities keep their order (the dependency order a
capture records), and the inherited checks (file sets, names, collapses, unreadable nodes) hold. --same BEFORE AFTER:
every file identical in records and order (ME10's two-device and one-device resident graphs).

usage: me8_router_nodes.py BEFORE_DIR AFTER_DIR      (ME8's check)
       me8_router_nodes.py --me10 BEFORE_DIR AFTER_DIR
       me8_router_nodes.py --same BEFORE_DIR AFTER_DIR
       me8_router_nodes.py --self-test                (CPU fixtures; no GPU)
"""
import collections
import os
import re
import shutil
import sys
import tempfile

GLOBAL_HASH = re.compile(r'(_GLOBAL__N__)[0-9a-f]{8}(?=_)')
NAME = re.compile(r'^graph-nodes-(?:(draft|catchup)-(?:sparse|dense)-)?(?:v\d+-)?B(\d+)-card(\d+)\.txt$')


def records(path, seq=None):
    c = collections.Counter()
    raw = collections.defaultdict(set)
    with open(path, errors='replace') as f:
        for line in f:
            parts = line.split(None, 1)
            if len(parts) != 2 or not parts[0].isdigit():
                raise ValueError('%s: an unparsed line: %r' % (path, line))
            r = parts[1].strip()
            n = GLOBAL_HASH.sub(r'\g<1>00000000', r)
            raw[n].add(r)
            c[n] += 1
            if seq is not None:
                seq.append(n)
    collapsed = sorted(n for n, rs in raw.items() if len(rs) > 1)
    return c, collapsed


def expected_removed(fname):
    m = NAME.match(fname)
    if m is None:
        return None
    kind, rows = m.group(1), int(m.group(2))
    return 48 if kind is None else rows if kind == 'draft' else 0


# ME10's changed identities: (pattern on a normalized record, its expected delta at B <= 3, at B 4 and 5, the launch
# shape an added record must have)
ME10_RULES = [
    (re.compile(r'^kernel \S*send_half_kernel\S* grid'), -96, -96, None),
    (re.compile(r'^kernel \S*recv_add_kernel\S* grid'), -96, -96, None),
    (re.compile(r'^kernel \S*combine_kernel\S* grid'), -48, -48, None),
    (re.compile(r'^kernel \S*finish_kernel\S* grid'), 48, 48, 'grid 10,1,1 block 256,1,1'),
    (re.compile(r'^kernel \S*gate_up_kernel_uI\S* grid'), 0, -192, None),
    (re.compile(r'^kernel \S*gate_up_kernel_u_pairI\S* grid'), 0, 96, 'grid 72,1,1 block 640,1,1'),
]


def file_sets(before, after, fails, need_kinds):
    fa = sorted(f for f in os.listdir(before) if f.startswith('graph-nodes'))
    fb = sorted(f for f in os.listdir(after) if f.startswith('graph-nodes'))
    if fa != fb:
        fails.append('file sets differ: only before %s, only after %s' %
                     (sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))))
    if not fa:
        fails.append('no graph listing')
    kinds = collections.defaultdict(set)
    for f in fa:
        m = NAME.match(f)
        if m is None:
            fails.append('%s: not a known graph listing name' % f)
            continue
        kinds[m.group(1) or 'target'].add(int(m.group(2)))
    for k in need_kinds:
        if not kinds[k]:
            fails.append('no %s graph listed' % k)
    if 'target' in need_kinds and kinds['target'] and \
            sorted(kinds['target']) != list(range(1, max(kinds['target']) + 1)):
        fails.append('target pass graphs not B1..B%d: %s' % (max(kinds['target']), sorted(kinds['target'])))
    if 'target' in need_kinds and kinds['target'] and max(kinds['target']) < 5:
        fails.append('no B 5 target graph (the pair at B 4 and 5 is unchecked)')
    return sorted(set(fa) & set(fb))


def check_me10(before, after, same=False, out=print):
    fails = []
    tag = 'me10 same' if same else 'me10'
    for f in file_sets(before, after, fails, () if same else ('target', 'draft', 'catchup')):
        m = NAME.match(f)
        if m is None:
            continue
        kind, rows = m.group(1) or 'target', int(m.group(2))
        sx, sy = [], []
        (x, cx), (y, cy) = records(os.path.join(before, f), sx), records(os.path.join(after, f), sy)
        bad = []
        for side, col in (('before', cx), ('after', cy)):
            if col:
                bad.append('%s: distinct records collapse under the name normalization: %s' % (side, col))
        delta = collections.Counter(y)
        delta.subtract(x)
        changed = {r: d for r, d in delta.items() if d != 0}
        rules = [] if same or kind != 'target' else ME10_RULES
        seen = collections.Counter()
        for r, d in changed.items():
            hit = [i for i, rule in enumerate(rules) if rule[0].search(r)]
            if len(hit) != 1:
                bad.append('an unexpected change: %+d of %s' % (d, r))
                continue
            seen[hit[0]] += d
            shape = rules[hit[0]][3]
            if d > 0 and shape is not None and not r.endswith(shape):
                bad.append('an added record of the wrong launch shape: %s' % r)
        for i, rule in enumerate(rules):
            want = rule[2] if rows >= 4 else rule[1]
            if seen[i] != want:
                bad.append('%s: %+d, expected %+d' % (rule[0].pattern, seen[i], want))
        keep = lambda seq: [r for r in seq if not any(rule[0].search(r) for rule in rules)]
        if keep(sx) != keep(sy):
            bad.append('the records outside the changed identities are not in the same order')
        for r in set(x) | set(y):
            if (r.startswith('type -1') or r.startswith('kernel ?')) and x[r] != y[r]:
                bad.append('an unreadable or unnamed node differs: %s' % r)
        out('%-52s %s' % (f, ('ok: %d records changed as expected' % sum(abs(d) for d in changed.values()))
                          if not bad else 'FAIL: ' + '; '.join(bad)))
        fails.extend('%s: %s' % (f, b) for b in bad)
    for x in fails:
        out('%s: FAIL: %s' % (tag, x))
    out('%s: %s (%d failures)' % (tag, 'PASS' if not fails else 'FAIL', len(fails)))
    return not fails


def check(before, after, out=print):
    fails = []
    fa = sorted(f for f in os.listdir(before) if f.startswith('graph-nodes'))
    fb = sorted(f for f in os.listdir(after) if f.startswith('graph-nodes'))
    if fa != fb:
        fails.append('file sets differ: only before %s, only after %s' %
                     (sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))))
    kinds = collections.defaultdict(set)
    for f in fa:
        m = NAME.match(f)
        if m is None:
            fails.append('%s: not a known graph listing name' % f)
            continue
        kinds[m.group(1) or 'target'].add(int(m.group(2)))
    for k in ('target', 'draft', 'catchup'):
        if not kinds[k]:
            fails.append('no %s graph listed' % k)
    if kinds['target'] and sorted(kinds['target']) != list(range(1, max(kinds['target']) + 1)):
        fails.append('target pass graphs not B1..B%d: %s' % (max(kinds['target']), sorted(kinds['target'])))
    for f in sorted(set(fa) & set(fb)):
        want = expected_removed(f)
        if want is None:
            continue
        (x, cx), (y, cy) = records(os.path.join(before, f)), records(os.path.join(after, f))
        added = y - x
        removed = x - y
        n_removed = sum(removed.values())
        ids = list(removed)
        bad = []
        for side, col in (('before', cx), ('after', cy)):
            if col:
                bad.append('%s: distinct records collapse under the name normalization: %s' % (side, col))
        if added:
            bad.append('adds %s' % dict(added))
        if n_removed != want:
            bad.append('removes %d nodes, expected %d' % (n_removed, want))
        if len(ids) > 1:
            bad.append('removes more than one identity: %s' % dict(removed))
        if len(ids) == 1 and not ids[0].startswith('kernel '):
            bad.append('removes a non-kernel node: %s' % ids[0])
        if len(ids) == 1 and y[ids[0]] < removed[ids[0]]:
            bad.append('fewer of the removed identity remain (%d) than were removed (%d): not one of two routers' %
                       (y[ids[0]], removed[ids[0]]))
        for r in set(x) | set(y):
            if (r.startswith('type -1') or r.startswith('kernel ?')) and x[r] != y[r]:
                bad.append('an unreadable or unnamed node differs: %s' % r)
        out('%-52s %s' % (f, 'ok: -%d of %s' % (n_removed, ids[0] if ids else '(nothing)') if not bad
                          else 'FAIL: ' + '; '.join(bad)))
        fails.extend('%s: %s' % (f, b) for b in bad)
    for x in fails:
        out('me8_router_nodes: FAIL: ' + x)
    out('me8_router_nodes: %s (%d files, %d failures)' % ('PASS' if not fails else 'FAIL', len(fa), len(fails)))
    return not fails


def self_test():
    router = 'kernel _ZN2fn5dense36_GLOBAL__N__%s_8_dense_cu_1a2b3c4d14product_kernelILi5ELb0EEEvNS0_7pargsE ' \
             'grid 72,1,1 block 256,1,1'
    attn = 'kernel _ZN2fn3qsa35_GLOBAL__N__%s_7_qsa_cu_5e6f7a8b11attn_kernelEv grid 12,1,1 block 128,1,1'
    plain = 'kernel gate_up_kernel grid 72,1,1 block 640,1,1'

    def graph(n_router, extra=(), h='4d5af096', attn_rec=None):
        a = attn_rec if attn_rec is not None else attn % h
        lines = [router % h] * n_router + [a] * 48 + [plain] * 96 + ['type 12 memory-operation'] * 20 + list(extra)
        return '\n'.join('%d %s' % (i, l) for i, l in enumerate(lines)) + '\n'

    def tree(root, routers_target, routers_draft, drop=(), extra=(), h='4d5af096', attn_rec=None):
        os.makedirs(root)
        for b in range(1, 6):
            if 'B%d' % b in drop:
                continue
            open(os.path.join(root, 'graph-nodes-B%d-card0.txt' % b), 'w').write(
                graph(routers_target, extra, h, attn_rec if b == 2 else None))
        for b in range(1, 5):
            open(os.path.join(root, 'graph-nodes-draft-sparse-v0-B%d-card0.txt' % b), 'w').write(
                graph(routers_draft(b), h=h))
            open(os.path.join(root, 'graph-nodes-catchup-sparse-v0-B%d-card0.txt' % b), 'w').write(graph(0, h=h))

    tmp = tempfile.mkdtemp()
    try:
        tree(os.path.join(tmp, 'before'), 96, lambda b: 2 * b)
        a2 = attn.replace('_7_qsa_cu_', '_7_qsb_cu_')
        a3 = attn.replace('11attn_kernelEv', '11attn_kernelIiEvv')
        cases = [('passing: 48 a target graph, one a proposal', dict(), True),
                 ('passing: another compilation component', dict(h='4db6c8e5'), True),
                 ('unchanged routers', dict(rt=96, rd=lambda b: 2 * b), False),
                 ('a missing B4 graph', dict(drop=('B4',)), False),
                 ('48 routers and an attention kernel removed', dict(attn=True), False),
                 ('an added node', dict(extra=('kernel extra grid 1,1,1 block 1,1,1',)), False),
                 ('every router removed (none of card 0 left)', dict(rt=0, rd=lambda b: 0), False),
                 ('a changed file component', dict(h='4db6c8e5', attn_rec=a2 % '4db6c8e5'), False),
                 ('a changed template', dict(attn_rec=a3 % '4d5af096'), False),
                 ('a changed grid', dict(attn_rec=(attn % '4d5af096').replace('grid 12,1,1', 'grid 13,1,1')), False),
                 ('a changed block', dict(attn_rec=(attn % '4d5af096').replace('block 128', 'block 64')), False),
                 ('two components collapse in one list', dict(extra=(attn % '0badf00d',)), False)]
        ok = True
        for i, (what, kw, want) in enumerate(cases):
            root = os.path.join(tmp, 'after%d' % i)
            tree(root, kw.get('rt', 48), kw.get('rd', lambda b: b), kw.get('drop', ()), kw.get('extra', ()),
                 kw.get('h', '4d5af096'), kw.get('attn_rec'))
            if kw.get('attn'):
                p = os.path.join(root, 'graph-nodes-B2-card0.txt')
                lines = open(p).read().splitlines()
                lines.remove(next(l for l in lines if 'attn_kernel' in l))
                open(p, 'w').write('\n'.join(lines) + '\n')
            got = check(os.path.join(tmp, 'before'), root, out=lambda s: None)
            print('me8_router_nodes self-test: %-46s %s (%s)' % (what, 'ok' if got == want else 'WRONG',
                                                                  'passes' if got else 'fails'))
            ok = ok and got == want
        ok = me10_self_test(tmp) and ok
        print('me8_router_nodes self-test: %s' % ('PASS' if ok else 'FAIL'))
        return ok
    finally:
        shutil.rmtree(tmp)


def me10_self_test(tmp):
    h = '4d5af096'
    gu = 'kernel _ZN2fn7experts36_GLOBAL__N__%s_8_experts_cu_1a2b3c4d16gate_up_kernel_uILb1ENS0_6gu_iq2ELi%dELi2EEE' \
         'vNS0_12gate_up_argsE grid 72,1,1 block 640,1,1'
    gp = 'kernel _ZN2fn7experts36_GLOBAL__N__%s_8_experts_cu_1a2b3c4d21gate_up_kernel_u_pairILb1ENS0_6gu_iq2ELi%dELi2' \
         'EEEvNS0_17gate_up_pair_argsE grid 72,1,1 block 640,1,1'
    sh = 'kernel _ZN2fn4xchg36_GLOBAL__N__%s_8_xchg_cu_5e6f7a8b16send_half_kernelEPKfiPji grid %d,1,1 block 128,1,1'
    ra = 'kernel _ZN2fn4xchg36_GLOBAL__N__%s_8_xchg_cu_5e6f7a8b15recv_add_kernelEPKfiPjPfS3_ji grid %d,1,1 ' \
         'block 128,1,1'
    cb = 'kernel _ZN2fn7experts36_GLOBAL__N__%s_8_cache_cu_9a8b7c6d14combine_kernelENS0_12combine_argsE grid 10,1,1 ' \
         'block 256,1,1'
    fi = 'kernel _ZN2fn7experts36_GLOBAL__N__%s_8_cache_cu_9a8b7c6d13finish_kernelENS0_12combine_argsEPKfPfS4_ grid ' \
         '10,1,1 block 256,1,1'
    other = 'kernel attn_kernel grid 12,1,1 block 128,1,1'

    def layer(b, me10, sabotage=''):
        g = (b + 3) // 4
        out = [other, 'type 12 memory-operation', sh % (h, g), sh % (h, g), ra % (h, g), ra % (h, g)]  # X-A
        if me10 and b >= 4 and sabotage != 'unpaired':
            out += [gp % (h, b)] * 2
        else:
            out += [gu % (h, b)] * 4
        if me10 and sabotage != 'no-finish':
            out += [fi % h]
        else:
            out += [cb % h, sh % (h, g), sh % (h, g), ra % (h, g), ra % (h, g)]  # X-F
        if sabotage == 'extra-gate-up' and me10:
            out += [gu % (h, b)]
        return out

    def write(root, me10, sabotage='', swap_order=False, h2=None):
        os.makedirs(root)
        for b in range(1, 6):
            lines = []
            for _ in range(48):
                lines += layer(b, me10, sabotage)
            if swap_order:
                lines[0], lines[1] = lines[1], lines[0]
            text = '\n'.join('%d %s' % (i, l) for i, l in enumerate(lines)) + '\n'
            if h2 is not None:
                text = text.replace(h, h2)
            open(os.path.join(root, 'graph-nodes-B%d-card0.txt' % b), 'w').write(text)
        for b in range(1, 5):
            for kind in ('draft', 'catchup'):
                open(os.path.join(root, 'graph-nodes-%s-sparse-v0-B%d-card0.txt' % (kind, b)), 'w').write(
                    '0 %s\n1 %s\n' % (other, sh % (h if h2 is None else h2, 1)))

    base = os.path.join(tmp, 'me10-before')
    write(base, False)
    cases = [('me10 passing: the finish, the pair at B 4 and 5', dict(), True, False),
             ('me10 passing: another compilation component', dict(h2='4db6c8e5'), True, False),
             ('me10 B 4 and 5 left unpaired', dict(sabotage='unpaired'), False, False),
             ('me10 X-F exchange kept', dict(sabotage='no-finish'), False, False),
             ('me10 an extra gate/up launch', dict(sabotage='extra-gate-up'), False, False),
             ('me10 another record reordered', dict(swap_order=True), False, False),
             ('same: identical lists', dict(me10=False), True, True),
             ('same: the ME10 change where none is expected', dict(), False, True)]
    ok = True
    for i, (what, kw, want, same) in enumerate(cases):
        root = os.path.join(tmp, 'me10-after%d' % i)
        write(root, kw.get('me10', True), kw.get('sabotage', ''), kw.get('swap_order', False), kw.get('h2'))
        got = check_me10(base, root, same, out=lambda s: None)
        print('me8_router_nodes self-test: %-46s %s (%s)' % (what, 'ok' if got == want else 'WRONG',
                                                              'passes' if got else 'fails'))
        ok = ok and got == want
    return ok


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] in ('--me10', '--same'):
        sys.exit(0 if check_me10(sys.argv[2], sys.argv[3], sys.argv[1] == '--same') else 1)
    if len(sys.argv) == 2 and sys.argv[1] == '--self-test':
        sys.exit(0 if self_test() else 1)
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(0 if check(sys.argv[1], sys.argv[2]) else 1)
