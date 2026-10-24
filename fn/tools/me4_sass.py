#!/usr/bin/env python3
"""ME4 (expires at ME4's disposition): compare me4_sass.sh's dumps (OUT_DIR/{base,me4}.sass, ptxas-*.txt).

Instructions are compared with addresses and encodings removed and the anonymous namespace's hash normalized.
PASS: every resident instantiation (X = 0), ep_group and ep_sum instruction-identical to BASE; every cache
instantiation's (X = 1) spill stores and loads no greater than BASE's; ep_task_off_kernel new, without spills.
"""
import re
import sys


def funcs(path):
    out, cur = {}, None
    for line in open(path):
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = re.sub(r"_GLOBAL__N__[0-9a-f]+_", "_GLOBAL__N__", m.group(1))
            out[cur] = []
            continue
        if cur and "/*" in line:
            ins = re.sub(r"/\*[0-9a-f]{4}\*/", "", line)
            ins = re.sub(r"/\* 0x[0-9a-f]+ \*/", "", ins).strip()
            if ins:
                out[cur].append(ins)
    return out


def spills(path):
    out, cur = {}, None
    for line in open(path):
        m = re.search(r"Function properties for (\S+)", line)
        if m:
            cur = re.sub(r"_GLOBAL__N__[0-9a-f]+_", "_GLOBAL__N__", m.group(1))
            continue
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and cur:
            out[cur] = (int(m.group(1)), int(m.group(2)))
            cur = None
    return out


def short(k):
    m = re.search(r"(ep_\w+?_kernel)(ILb([01])ELNS0_3fmtE(\d+)E)?", k)
    if not m:
        return k
    return m.group(1) + ("<X=%s,fmt=%s>" % (m.group(3), m.group(4)) if m.group(2) else "")


def main():
    d = sys.argv[1]
    a, b = funcs(d + "/base.sass"), funcs(d + "/me4.sass")
    sa, sb = spills(d + "/ptxas-base.txt"), spills(d + "/ptxas-me4.txt")
    ok = True
    for k in sorted(set(a) | set(b)):
        n = short(k)
        sp = sb.get(k, (0, 0))
        if k not in a:
            good = "ep_task_off_kernel" in k and sp == (0, 0)
            print("new       %-28s %6d instructions, spills %s: %s" % (n, len(b[k]), sp, "ok" if good else "FAIL"))
        elif k not in b:
            good = False
            print("removed   %-28s FAIL" % n)
        elif "<X=1" in n:
            base_sp = sa.get(k, (0, 0))
            good = sp[0] <= base_sp[0] and sp[1] <= base_sp[1]
            print("cache     %-28s %6d -> %6d instructions, spills %s (base %s): %s" %
                  (n, len(a[k]), len(b[k]), sp, base_sp, "ok" if good else "FAIL"))
        else:
            good = a[k] == b[k]
            print("resident  %-28s %6d -> %6d instructions: %s" % (n, len(a[k]), len(b[k]),
                                                                 "identical" if good else "DIFFERS (FAIL)"))
        ok = ok and good
    print("ME4 SASS check:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
