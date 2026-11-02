#!/usr/bin/env python3
"""TTTS3 Amendment 1 line 2: hidden costs in one object's SASS (static, no GPU).

For each function of a cuobjdump -sass dump: its CALL, RET, LDL and STL instructions, and its stack frame from the
matching cuobjdump -res-usage (STACK, LOCAL). Prints the functions with any nonzero count and a total line; with
--base BEFORE.sass BEFORE.res, also marks which of them are unchanged from the base (present at the base with the same
counts).

usage: ttts3_hidden_cost.py OBJ.sass OBJ.res [--base BEFORE.sass BEFORE.res]
"""
import re
import sys

GLOBAL_HASH = re.compile(r'(_GLOBAL__N__)[0-9a-f]{8}(?=_)')
FUNC = re.compile(r'^\s*Function : (\S+)')
INSN = re.compile(r'^\s*/\*[0-9a-f]{4,}\*/\s*(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)')
RES = re.compile(r'Function (\S+):')
KINDS = ('CALL', 'RET', 'LDL', 'STL')


def counts(sass, res):
    out, name = {}, None
    for line in open(sass, errors='replace'):
        m = FUNC.match(line)
        if m:
            name = GLOBAL_HASH.sub(r'\g<1>00000000', m.group(1))
            out[name] = dict.fromkeys(KINDS + ('STACK', 'LOCAL'), 0)
            continue
        m = INSN.match(line)
        if m and name:
            op = m.group(1).split('.')[0]
            if op in KINDS:
                out[name][op] += 1
    name = None
    for line in open(res, errors='replace'):
        m = RES.search(line)
        if m:
            name = GLOBAL_HASH.sub(r'\g<1>00000000', m.group(1))
            continue
        if name and 'REG:' in line and name in out:
            for k in ('STACK', 'LOCAL'):
                mm = re.search(k + r':(\d+)', line)
                out[name][k] = int(mm.group(1)) if mm else 0
            name = None
    return out


def main(argv):
    if len(argv) not in (3, 6):
        sys.exit(__doc__)
    c = counts(argv[1], argv[2])
    base = counts(argv[4], argv[5]) if len(argv) == 6 else None
    tot = dict.fromkeys(KINDS + ('STACK',), 0)
    n_bad = n_new_bad = 0
    for n, v in c.items():
        if any(v[k] for k in KINDS + ('STACK', 'LOCAL')):
            n_bad += 1
            tag = ''
            if base is not None:
                same = n in base and base[n] == v
                tag = ' (as at the base)' if same else ' (NOT at the base)'
                n_new_bad += 0 if same else 1
            print('%s: CALL %d RET %d LDL %d STL %d STACK %d LOCAL %d%s' %
                  (n, v['CALL'], v['RET'], v['LDL'], v['STL'], v['STACK'], v['LOCAL'], tag))
        for k in tot:
            tot[k] += v[k]
    print('ttts3_hidden_cost: %d functions, %d with a nonzero count%s; totals CALL %d RET %d LDL %d STL %d' %
          (len(c), n_bad, '' if base is None else ' (%d not as at the base)' % n_new_bad, tot['CALL'], tot['RET'],
           tot['LDL'], tot['STL']))


if __name__ == '__main__':
    main(sys.argv)
