#!/usr/bin/env python3
"""ME10: classify the SASS changes of functions me10_sass_same.py reports CHANGED (static, no GPU).

usage: me10_sass_classify.py BEFORE.sass AFTER.sass BEFORE.res AFTER.res [NAME_SUBSTRING]

For each function present in both and different: resources from cuobjdump -res-usage (REG, STACK, SHARED, LOCAL), the
instruction count, and three views of the difference:
  relocation-only: equal once branch/call targets (absolute code addresses) and NOPs are dropped;
  the opcode histogram's difference (instruction kinds added or removed);
  difflib's first changed hunks on the address-free instruction text, with their positions.
Nothing is normalized beyond addresses and NOPs: registers, arithmetic, scheduling and branches stay compared.
"""
import collections
import difflib
import re
import sys

GLOBAL_HASH = re.compile(r'(_GLOBAL__N__)[0-9a-f]{8}(?=_)')
FUNC = re.compile(r'^\s*Function : (\S+)')
INSN = re.compile(r'^\s*/\*([0-9a-f]{4,})\*/\s*(.*?)\s*;')
RES = re.compile(r'Function (\S+):')
TARGET = re.compile(r'\b(BRA|BRX|CALL\.\S+|CALL|JMP|BSSY|SSY|PBK|BREAK|SYNC|WARPSYNC|BMOV)\b(.*)')


def functions(path):
    out, name = {}, None
    with open(path, errors='replace') as f:
        for line in f:
            m = FUNC.match(line)
            if m:
                name = GLOBAL_HASH.sub(r'\g<1>00000000', m.group(1))
                out[name] = []
                continue
            m = INSN.match(line)
            if m and name is not None:
                out[name].append(GLOBAL_HASH.sub(r'\g<1>00000000', m.group(2)))
    return out


def resources(path):
    out, name = {}, None
    for line in open(path, errors='replace'):
        m = RES.search(line)
        if m:
            name = GLOBAL_HASH.sub(r'\g<1>00000000', m.group(1))
            continue
        if name and 'REG:' in line:
            out[name] = ' '.join(x for x in line.split() if re.match(r'(REG|STACK|SHARED|LOCAL):', x))
            name = None
    return out


def no_addr(i):
    # code addresses in branch and reconvergence operands: 0x... immediates after those opcodes
    if TARGET.search(i):
        return re.sub(r'0x[0-9a-f]+', '<addr>', i)
    return i


def opcode(i):
    i = re.sub(r'^@!?U?P[T0-9]+\s+', '', i)
    return i.split()[0] if i.split() else ''


def main():
    bs, as_, br, ar = sys.argv[1:5]
    pick = sys.argv[5] if len(sys.argv) > 5 else ''
    a, b = functions(bs), functions(as_)
    ra, rb = resources(br), resources(ar)
    n_meaning = n_reloc = 0
    for n in a:
        if n not in b or a[n] == b[n] or pick not in n:
            continue
        x = [no_addr(i) for i in a[n] if opcode(i) != 'NOP']
        y = [no_addr(i) for i in b[n] if opcode(i) != 'NOP']
        reloc = x == y
        n_reloc += reloc
        n_meaning += not reloc
        short = re.sub(r'^.*?(gate_up_kernel_u\w*?I)', r'\1', n)
        print('== %s' % short)
        print('   resources: before %s | after %s%s' % (ra.get(n, '?'), rb.get(n, '?'),
                                                       '' if ra.get(n) == rb.get(n) else '  (CHANGED)'))
        nops = (sum(opcode(i) == 'NOP' for i in a[n]), sum(opcode(i) == 'NOP' for i in b[n]))
        print('   instructions %d -> %d (NOP %d -> %d); without NOPs and code addresses: %s' %
              (len(a[n]), len(b[n]), nops[0], nops[1], 'EQUAL (relocation/alignment only)' if reloc else 'differ'))
        if reloc:
            continue
        ha, hb = collections.Counter(map(opcode, x)), collections.Counter(map(opcode, y))
        d = {k: hb[k] - ha[k] for k in set(ha) | set(hb) if hb[k] != ha[k]}
        print('   opcode histogram change: %s' % (', '.join('%s %+d' % kv for kv in sorted(d.items())) or
                                                  'none (same instructions, different order or operands)'))
        sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
        ops = [o for o in sm.get_opcodes() if o[0] != 'equal']
        changed_lines = sum(max(o[2] - o[1], o[4] - o[3]) for o in ops)
        print('   %d differing hunks, %d instructions touched' % (len(ops), changed_lines))
        for tag, i1, i2, j1, j2 in ops[:4]:
            print('   hunk %s before[%d:%d] after[%d:%d]' % (tag, i1, i2, j1, j2))
            for i in x[i1:min(i2, i1 + 4)]:
                print('     - %s' % i)
            for i in y[j1:min(j2, j1 + 4)]:
                print('     + %s' % i)
    print('me10_sass_classify: %d changed functions are relocation/alignment only, %d differ in instructions' %
          (n_reloc, n_meaning))


if __name__ == '__main__':
    main()
