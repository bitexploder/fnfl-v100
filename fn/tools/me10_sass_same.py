#!/usr/bin/env python3
"""ME10: compare two builds' SASS of one object (cuobjdump -sass output, BEFORE then AFTER), function by function.

Each function's instructions are read without their addresses and encodings; a name's per-compilation component
(the eight hex digits after _GLOBAL__N__, me8_router_nodes.py's rule) is normalized. Reports every function of
BEFORE that AFTER holds unchanged, changed or missing, and AFTER's new functions with their instruction counts beside
the BEFORE function they pair with (gate_up_kernel_u_pair against gate_up_kernel_u of the same template arguments).
Exit 0 when every BEFORE function is in AFTER unchanged and AFTER adds only ME10's kernels (gate_up_kernel_u_pair,
finish_kernel).

usage: me10_sass_same.py BEFORE.sass AFTER.sass
"""
import re
import sys

GLOBAL_HASH = re.compile(r'(_GLOBAL__N__)[0-9a-f]{8}(?=_)')
FUNC = re.compile(r'^\s*Function : (\S+)')
INSN = re.compile(r'^\s*/\*[0-9a-f]{4,}\*/\s*(.*?)\s*;')


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
                out[name].append(GLOBAL_HASH.sub(r'\g<1>00000000', m.group(1)))
    return out


def main(before, after):
    a, b = functions(before), functions(after)
    same = [n for n in a if n in b and a[n] == b[n]]
    changed = [n for n in a if n in b and a[n] != b[n]]
    missing = [n for n in a if n not in b]
    new = [n for n in b if n not in a]
    for n in changed:
        print('CHANGED %s: %d -> %d instructions' % (n, len(a[n]), len(b[n])))
    for n in missing:
        print('MISSING %s' % n)
    bad_new = []
    for n in new:
        base = n.replace('21gate_up_kernel_u_pairI', '16gate_up_kernel_uI').replace('NS0_17gate_up_pair_argsE',
                                                                                  'NS0_12gate_up_argsE')
        if 'gate_up_kernel_u_pair' not in n and 'finish_kernel' not in n:
            bad_new.append(n)
        ref = len(a[base]) if base in a else -1
        print('NEW %s: %d instructions (gate_up_kernel_u of the same arguments: %d)' % (n, len(b[n]), ref))
    ok = not changed and not missing and not bad_new and bool(a)
    print('me10_sass_same: %d functions of BEFORE unchanged, %d changed, %d missing; %d new: %s' %
          (len(same), len(changed), len(missing), len(new), 'PASS' if ok else 'FAIL'))
    return ok


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
