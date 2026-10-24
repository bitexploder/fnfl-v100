#!/usr/bin/env python3
"""ME11: the GDN replay's FMUL/FFMA data dependence, read from SASS (a static check, no GPU).

Usage: me11_sass_replay.py <cuobjdump -sass of gdn.cu.o> <cuobjdump -sass of gdn_prompt.cu.o>
       me11_sass_replay.py --same <SASS of a pre-ME11 gdn.cu.o> <SASS of gdn_ref.cu.o>
       me11_sass_replay.py --self-test

--same: the test reference (gdn_ref.cu's core_kernel<B>) against the pre-ME11 production gdn_core<B>, instruction by
instruction (addresses dropped), B = 1..5: the reference the bitwise checks use is the original kernel's code.

In every gdn_core<B> instance (B = 1..5) and in the prompt scan, find the replay loop: a loop body (a backward branch
and its target) holding FMULs that share one source register (the decay), each product read first by an FFMA as its
addend (the third source), nowhere else before. gdn_core needs 4 such pairs a loop (a lane's float4), the scan 16 (a
thread's cells). Any FMUL of the shared register whose product is read otherwise, or an FFMA taking the decay as a
multiplicand (a contraction), fails. Exit 0 for PASS.
"""

import re
import sys

INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?P\d\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);")
FUN = re.compile(r"Function : (\S+)")


def functions(text):
    out, name = {}, None
    for line in text.splitlines():
        m = FUN.search(line)
        if m:
            name = m.group(1)
            out[name] = []
            continue
        m = INS.search(line)
        if m and name is not None:
            ops = [o.strip() for o in m.group(4).split(",")] if m.group(4).strip() else []
            out[name].append((int(m.group(1), 16), m.group(3), ops))
    return out


def reg(o):
    m = re.match(r"-?\|?(R\d+)", o)
    return m.group(1) if m else None


def loops(body):
    for i, (addr, op, ops) in enumerate(body):
        if op.startswith("BRA") and ops:
            try:
                target = int(ops[-1].split()[0], 16)
            except ValueError:
                continue
            if target <= addr:
                yield [ins for ins in body if target <= ins[0] <= addr]


def pairs_in(loop):
    """(pairs, problems) for the most shared FMUL source register in the loop"""
    fmuls = [(k, ins) for k, ins in enumerate(loop) if ins[1] == "FMUL" and len(ins[2]) == 3]
    count = {}
    for _, ins in fmuls:
        for o in ins[2][1:]:
            r = reg(o)
            if r:
                count[r] = count.get(r, 0) + 1
    if not count:
        return 0, []
    dec = max(count, key=count.get)
    pairs, problems = 0, []
    for k, ins in fmuls:
        if dec not in [reg(o) for o in ins[2][1:]]:
            continue
        dst = reg(ins[2][0])
        for later in loop[k + 1:]:
            srcs = [reg(o) for o in later[2][1:]]
            if dst in srcs:
                if later[1] == "FFMA" and len(srcs) == 3 and srcs[2] == dst and dst not in srcs[:2]:
                    pairs += 1
                else:
                    problems.append("%04x %s %s reads the product %s" % (later[0], later[1], ",".join(later[2]), dst))
                break
            if later[2] and reg(later[2][0]) == dst:
                problems.append("%04x: the product %s is overwritten unread" % (ins[0], dst))
                break
    for ins in loop:
        if ins[1] == "FFMA" and len(ins[2]) == 4 and dec in [reg(o) for o in ins[2][1:3]]:
            problems.append("%04x FFMA takes the decay %s as a multiplicand (contraction)" % (ins[0], dec))
    return pairs, problems


def check(fns, pattern, want, label):
    names = [n for n in fns if pattern in n]
    ok = bool(names)
    for n in names:
        best, why = 0, []
        for lp in loops(fns[n]):
            p, probs = pairs_in(lp)
            if p == want and not probs:
                best, why = p, []
                break
            if p > best:
                best, why = p, probs
        good = best == want and not why
        ok = ok and good
        print("%s %s: replay loop FMUL -> FFMA addend pairs %d (want %d)%s" %
              (label, n[-60:], best, want, "" if good else ": FAIL " + "; ".join(why)))
    if not names:
        print("%s: no function matching %s" % (label, pattern))
    return ok


def self_test():
    good = """
        Function : core
        /*0010*/ LDG.E.CONSTANT.SYS R2, [R2] ;
        /*0020*/ FMUL R29, R2, R8 ;
        /*0030*/ FMUL R9, R2, R9 ;
        /*0040*/ FMUL R31, R2, R10 ;
        /*0050*/ FMUL R11, R2, R11 ;
        /*0060*/ FFMA R8, R12, R24, R29 ;
        /*0070*/ FFMA R9, R13, R24, R9 ;
        /*0080*/ FFMA R10, R14, R24, R31 ;
        /*0090*/ FFMA R11, R15, R24, R11 ;
        /*00a0*/ @!P1 BRA 0x10 ;
    """
    bad = good.replace("FFMA R8, R12, R24, R29", "FADD R8, R12, R29").replace("Function : core", "Function : core")
    fused = good.replace("FMUL R29, R2, R8 ;", "NOP ;").replace("FFMA R8, R12, R24, R29", "FFMA R8, R2, R8, R12")
    a = functions(good)
    b = functions(bad)
    c = functions(fused)
    ok = (pairs_in(next(loops(a["core"]))) == (4, []) and pairs_in(next(loops(b["core"])))[1] != [] and
          pairs_in(next(loops(c["core"])))[1] != [])
    print("me11_sass_replay self-test: %s" % ("PASS" if ok else "FAIL"))
    return ok


def same(base_path, ref_path):
    def bodies(path):
        out, name = {}, None
        for line in open(path):
            m = FUN.search(line)
            if m:
                name = m.group(1)
                out[name] = []
                continue
            m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(.*?);", line)
            if m and name is not None:
                out[name].append(re.sub(r"\s+", " ", m.group(1)).strip())
        return out
    a, b = bodies(base_path), bodies(ref_path)
    ok = True
    for B in range(1, 6):
        ka = [n for n in a if "core_kernelILi%dE" % B in n]
        kb = [n for n in b if "core_kernelILi%dE" % B in n and "core_ref_args" in n]
        good = len(ka) == 1 and len(kb) == 1 and a[ka[0]] == b[kb[0]]
        ok = ok and good
        print("core_kernel<%d>: pre-ME11 %d instructions, gdn_ref %d: %s" %
              (B, len(a[ka[0]]) if ka else 0, len(b[kb[0]]) if kb else 0, "identical" if good else "DIFFERENT"))
    print("me11_sass_replay same: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--same":
        return same(sys.argv[2], sys.argv[3])
    if sys.argv[1:] == ["--self-test"]:
        return 0 if self_test() else 1
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    core = functions(open(sys.argv[1]).read())
    scan = functions(open(sys.argv[2]).read())
    ok = True
    for b in range(1, 6):
        ok = check(core, "core_kernelILi%dE" % b, 4, "gdn_core<%d>" % b) and ok
    ok = check(scan, "scan_kernel", 16, "scan") and ok
    print("me11_sass_replay: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
