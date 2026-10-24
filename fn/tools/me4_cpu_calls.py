#!/usr/bin/env python3
"""ME4 (expires at ME4's disposition): the CPU miss path's compiled call sites. Reads `objdump -dr -C` of the object
that holds fn::experts::cpu::pool::drain and checks, on its control-flow graph, that the native rows' branch has no
layout reconstruction (Shell's addition: "inspect compiled CPU call sites").

In drain a unit's work is one loop iteration, and every path out of it is a backward jump: to the loop head, or (with
loop rotation, as -O3 compiles it) straight into the next unit's body. So a unit's reach ends at any backward-jump
target. For each null test (`test %r,%r` then je/jne) the two successors are followed forward, without crossing a
backward-jump target, collecting the calls each reaches. A native branch is a successor that reaches a product (gate_up_tile or
down_rows) and no un-permute (unpack_gate_up_tile, unpack_down_rows, unpack_down_card), while the other successor
reaches the un-permute. Pass: such a branch for gate/up and one for down. A product is the wrapper (gate_up_tile,
down_rows) or, where it is inlined, a row kernel it calls (iq_gu_rows, kq_gu_rows; iq4nl_down_rows, kq_down_rows,
q2_down_rows).

usage: me4_cpu_calls.py OBJECT   (exits 1 on a failed check; prints the call sites and branches)
"""
import re
import subprocess
import sys

# the products: the wrappers, or (where the compiler inlines them, e.g. -O3 without PIC) the row kernels they call
PRODUCT = {"gate_up_tile(": "gate_up_tile", "iq_gu_rows(": "gate_up_tile", "kq_gu_rows(": "gate_up_tile",
           "down_rows(": "down_rows", "iq4nl_down_rows(": "down_rows", "kq_down_rows(": "down_rows",
           "q2_down_rows(": "down_rows"}
UNPACK = ("unpack_gate_up_tile(", "unpack_down_rows(", "unpack_down_card(")


def disasm(obj):
    out = subprocess.run(["objdump", "-dr", "--no-show-raw-insn", "-C", obj], check=True, capture_output=True,
                         text=True).stdout
    insns, cur, keep = [], None, False
    for line in out.split("\n"):
        m = re.match(r"^([0-9a-f]+) <(.*)>:$", line)
        if m:
            keep = "pool::drain(" in m.group(2)
            continue
        if not keep:
            continue
        m = re.match(r"^\s+[0-9a-f]+: R_X86_64_\w+\s+(.*)$", line)
        if m:
            if cur is not None:
                cur["reloc"] = m.group(1)
            continue
        m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", line)
        if m:
            cur = {"a": int(m.group(1), 16), "op": m.group(2), "arg": m.group(3), "reloc": ""}
            insns.append(cur)
    return insns


def callee(i):
    s = i["reloc"] or i["arg"]
    for name in UNPACK:
        if name in s:
            return "unpack:" + name[:-1]
    for name, kind in PRODUCT.items():
        if re.search(r"(^|::)" + re.escape(name), s):
            return "product:" + kind
    return None


def main():
    insns = disasm(sys.argv[1])
    if not insns:
        sys.exit("me4_cpu_calls: no pool::drain in " + sys.argv[1])
    idx = {i["a"]: k for k, i in enumerate(insns)}

    def target(i):
        m = re.match(r"^([0-9a-f]+)\b", i["arg"])
        return int(m.group(1), 16) if m else None

    back = {}
    for i in insns:
        if i["op"].startswith("j"):
            t = target(i)
            if t is not None and t < i["a"] and t in idx:
                back[t] = back.get(t, 0) + 1
    head = max(back, key=back.get) if back else None
    stops = set(back)  # every backward-jump target: a unit's work ends there

    def succ(k):
        i = insns[k]
        op = i["op"]
        out = []
        if op.startswith("j"):
            t = target(i)
            if t is not None and t in idx:
                out.append(idx[t])
            if op != "jmp" and k + 1 < len(insns):
                out.append(k + 1)
        elif op.startswith("ret") or op == "ud2":
            pass
        elif k + 1 < len(insns):
            out.append(k + 1)
        return out

    def reach(k0):
        seen, stack, calls = set(), [k0], set()
        while stack:
            k = stack.pop()
            if k in seen or (insns[k]["a"] in stops and k != k0):
                continue
            seen.add(k)
            if insns[k]["op"].startswith("call"):
                c = callee(insns[k])
                if c:
                    calls.add(c)
            stack.extend(succ(k))
        return calls

    sites = [(i["a"], callee(i)) for i in insns if i["op"].startswith("call") and callee(i)]
    print("me4_cpu_calls: pool::drain, %d instructions, loop head 0x%x (%d backward jumps); %d backward-jump targets "
          "end a unit's reach: %s" % (len(insns), head or 0, back.get(head, 0), len(stops),
                                       " ".join("0x%x" % a for a in sorted(stops))))
    for a, c in sites:
        print("  call site 0x%x: %s" % (a, c))
    native = {"gate_up_tile": False, "down_rows": False}
    for k in range(len(insns) - 1):
        i, j = insns[k], insns[k + 1]
        if i["op"] != "test" or j["op"] not in ("je", "jne"):
            continue
        r = i["arg"].split(",")
        if len(r) != 2 or r[0] != r[1]:
            continue
        t = target(j)
        if t is None or t not in idx:
            continue
        taken, fall = reach(idx[t]), reach(k + 2)
        for nat, ref, how in ((fall, taken, "fall-through"), (taken, fall, "taken")):
            prods = {c.split(":")[1] for c in nat if c.startswith("product:")}
            if prods and not any(c.startswith("unpack:") for c in nat) and any(c.startswith("unpack:") for c in ref):
                for p in prods & set(native):
                    native[p] = True
                print("  native branch at 0x%x (%s %s, %s): reaches %s; no un-permute (the other successor reaches %s)"
                      % (i["a"], j["op"], i["arg"], how, ", ".join(sorted(nat)),
                         ", ".join(sorted(c for c in ref if c.startswith("unpack:")))))
    ok = all(native.values())
    print("me4_cpu_calls: native gate/up branch %s, native down branch %s: %s" %
          ("found" if native["gate_up_tile"] else "MISSING", "found" if native["down_rows"] else "MISSING",
           "PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
