#!/usr/bin/env python3
"""TTTS3 Amendment 1 line 1: ME10's per-function SASS and resource comparison with a rename map (static, no GPU).

Like me10_sass_same.py, but a function of BEFORE pairs with the function of AFTER whose demangled name is the same
once the rename map is applied to AFTER's demangled names (TTTS3 item 2: gu_iq2 and gu_iq3 became aliases of
gu_family<...>, so the IQ bodies' mangled names changed). Each pair compares the instructions without addresses or
encodings (ME10's rule) and the -res-usage line (REG, STACK, SHARED, LOCAL). Prints the old -> new name map of every
renamed function, each CHANGED or MISSING function of BEFORE and AFTER's NEW functions (listed, not failing: the
items' new kernels). Exit 0 when every BEFORE function pairs with an identical AFTER function of identical resources.

usage: ttts3_sass_map.py BEFORE.sass AFTER.sass BEFORE.res AFTER.res
"""
import re
import subprocess
import sys

GLOBAL_HASH = re.compile(r'(_GLOBAL__N__)[0-9a-f]{8}(?=_)')
FUNC = re.compile(r'^\s*Function : (\S+)')
INSN = re.compile(r'^\s*/\*[0-9a-f]{4,}\*/\s*(.*?)\s*;')
RES = re.compile(r'Function (\S+):')
F = r'\(fn::experts::fmt\)'
# AFTER's demangled text -> BEFORE's
RENAME = [
    (re.compile(r'gu_family<%s0, %s1, %s2>' % (F, F, F)), 'gu_iq2'),
    (re.compile(r'gu_family<%s3, %s4>' % (F, F)), 'gu_iq3'),
]


def norm(n):
    return GLOBAL_HASH.sub(r'\g<1>00000000', n)


def functions(path):
    out, name = {}, None
    with open(path, errors='replace') as f:
        for line in f:
            m = FUNC.match(line)
            if m:
                name = norm(m.group(1))
                out[name] = []
                continue
            m = INSN.match(line)
            if m and name is not None:
                out[name].append(norm(m.group(1)))
    return out


def resources(path):
    out, name = {}, None
    for line in open(path, errors='replace'):
        m = RES.search(line)
        if m:
            name = norm(m.group(1))
            continue
        if name and 'REG:' in line:
            out[name] = ' '.join(x for x in line.split() if re.match(r'(REG|STACK|SHARED|LOCAL):', x))
            name = None
    return out


def demangle(names):
    names = list(names)
    r = subprocess.run(['c++filt'], input='\n'.join(names) + '\n', capture_output=True, text=True, check=True)
    return dict(zip(names, r.stdout.split('\n')))


def main(bs, as_, br, ar):
    a, b = functions(bs), functions(as_)
    ra, rb = resources(br), resources(ar)
    da, db = demangle(a), demangle(b)
    key_b = {}
    for n, d in db.items():
        for pat, rep in RENAME:
            d = pat.sub(rep, d)
        key_b[d] = n
    same = changed = missing = 0
    renamed, paired = [], set()
    for n in a:
        m = key_b.get(da[n])
        if m is None:
            print('MISSING %s' % n)
            missing += 1
            continue
        paired.add(m)
        if m != n:
            renamed.append((n, m))
        if a[n] != b[m] or ra.get(n) != rb.get(m):
            print('CHANGED %s: %d -> %d instructions; %s -> %s' % (n, len(a[n]), len(b[m]), ra.get(n), rb.get(m)))
            changed += 1
        else:
            same += 1
    for n, m in renamed:
        print('MAP %s -> %s' % (n, m))
    new = [n for n in b if n not in paired]
    for n in new:
        print('NEW %s: %d instructions; %s' % (n, len(b[n]), rb.get(n)))
    ok = bool(a) and not changed and not missing
    print('ttts3_sass_map: %d functions of BEFORE identical (%d renamed), %d changed, %d missing; %d new: %s' %
          (same, len(renamed), changed, missing, len(new), 'PASS' if ok else 'FAIL'))
    return ok


if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    sys.exit(0 if main(*sys.argv[1:]) else 1)
