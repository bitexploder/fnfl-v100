#!/usr/bin/env python3
"""ME11: two builds' captured graphs, node for node (a static comparison of model-dry's files, no GPU).

Usage: me11_graph_same.py BASE_DIR ME11_DIR
       me11_graph_same.py --self-test

For every graph-nodes-*.txt in BASE_DIR (target, draft and catch-up graphs, model-dry's graph mode), the file of the
same name in ME11_DIR must list the same nodes in the same order: type, kernel, grid and block. ME11 changes the GDN
kernels' arguments, not their launches, so nothing is added or removed.

Kernel names are compared exactly after one normalization (Fable's approval, lab 75e0528a; the rule ME8's checker
uses): the eight hex digits after "_GLOBAL__N__", the anonymous namespace's source hash, which differs between two
builds of the same file. The kernel family, its template arguments and argument types, and every launch field are
kept. RENAMES lists any documented argument-type rename; ME11 renames none (core_args, prompt_args, commit_args and
finish_args keep their names), so it is empty. A changed B or a changed kernel fails.

The graph-edges-*.txt beside each must hold the same dependencies when both builds wrote them (ME11's build does). A
base without them leaves the dependencies uncompared, reported per graph and in the summary; never as equal. Exit 0
for PASS of the nodes and of every compared set of edges.
"""

import os
import re
import sys

GLOBAL_HASH = re.compile(r"(_GLOBAL__N__)[0-9a-f]{8}(?=_)")
# documented argument-type renames, (base substring, ME11 substring); ME11 has none
RENAMES = []


def norm(name):
    name = GLOBAL_HASH.sub(r"\g<1>00000000", name)
    for old, new in RENAMES:
        name = name.replace(old, new)
    return name


def nodes(path):
    out = []
    for line in open(path):
        parts = line.split()
        if len(parts) >= 2:
            out.append(parts[1:])
    return out


def compare(base, cand):
    """(ok, renamed, why): renamed counts kernel nodes equal only after the normalization"""
    if len(base) != len(cand):
        return False, 0, "node count %d against %d" % (len(base), len(cand))
    renamed = 0
    for i, (a, b) in enumerate(zip(base, cand)):
        if a == b:
            continue
        if a[0] == "kernel" and b[0] == "kernel" and len(a) == len(b) and a[2:] == b[2:] and norm(a[1]) == norm(b[1]):
            renamed += 1
            continue
        return False, renamed, "node %d: %s against %s" % (i, " ".join(a), " ".join(b))
    return True, renamed, ""


def edges(path):
    if not os.path.exists(path):
        return None
    return sorted(tuple(line.split()) for line in open(path) if line.strip())


def main():
    if sys.argv[1:] == ["--self-test"]:
        def k(name, grid="144,1,1", block="704,1,1"):
            return ["kernel", name, "grid", grid, "block", block]
        core = "_ZN2fn3gdn38_GLOBAL__N__%s_6_gdn_cu_5786f24a11core_kernelILi%dEEEvNS0_9core_argsE"
        fin = "_ZN2fn5round40_GLOBAL__N__%s_8_round_cu_973253cc13finish_kernelENS0_11finish_argsE"
        a = [k("x", "1,1,1", "32,1,1"), ["type", "1", "memcpy"]]
        cases = [
            ("identical lists", compare(a, a) == (True, 0, "")),
            ("another kernel fails", not compare(a, [k("y", "1,1,1", "32,1,1"), a[1]])[0]),
            ("a missing node fails", not compare(a, a[:1])[0]),
            ("the source-hash rename passes", compare([k(core % ("55378419", 5))], [k(core % ("0bad7311", 5))]) ==
             (True, 1, "")),
            ("a changed B fails (core_kernel<1> -> <5>)",
             not compare([k(core % ("55378419", 1))], [k(core % ("55378419", 5))])[0]),
            ("a changed B under a hash change fails", not compare([k(core % ("55378419", 1))],
                                                                  [k(core % ("0bad7311", 5))])[0]),
            ("a changed family fails (core_kernel -> finish_kernel)",
             not compare([k(core % ("55378419", 1))], [k(fin % "55378419")])[0]),
            ("a changed launch field fails", not compare([k(core % ("55378419", 5))],
                                                         [k(core % ("55378419", 5), "72,1,1")])[0]),
            ("a changed file component fails", not compare([k(core % ("55378419", 5))],
                                                           [k((core % ("55378419", 5)).replace("_6_gdn_cu_",
                                                                                                "_6_gdx_cu_"))])[0]),
        ]
        for name, good in cases:
            print("me11_graph_same self-test: %s: %s" % (name, "ok" if good else "FAIL"))
        ok = all(good for _, good in cases)
        print("me11_graph_same self-test: %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    base_dir, cand_dir = sys.argv[1], sys.argv[2]
    files = sorted(f for f in os.listdir(base_dir) if f.startswith("graph-nodes-") and f.endswith(".txt"))
    ok = bool(files)
    total = 0
    edges_compared = edges_uncompared = 0
    for f in files:
        cp = os.path.join(cand_dir, f)
        if not os.path.exists(cp):
            print("%s: missing in %s: FAIL" % (f, cand_dir))
            ok = False
            continue
        bn, cn = nodes(os.path.join(base_dir, f)), nodes(cp)
        good, renamed, why = compare(bn, cn)
        e = f.replace("graph-nodes-", "graph-edges-", 1)
        be, ce = edges(os.path.join(base_dir, e)), edges(os.path.join(cand_dir, e))
        if ce is None:
            dep = "ME11 edges missing: FAIL"
            good = False
        elif be is None:
            dep = "%d edges, UNCOMPARED (the base wrote none)" % len(ce)
            edges_uncompared += 1
        elif be == ce:
            dep = "%d edges, the same" % len(ce)
            edges_compared += 1
        else:
            dep = "edges DIFFER (%d against %d)" % (len(be), len(ce))
            edges_compared += 1
            good = False
        total += len(cn)
        print("%s: %d nodes %s%s; %s" % (f, len(cn), "DIFFER: " + why if why else "the same",
                                        ", %d kernel names equal after the source-hash normalization" % renamed
                                        if renamed else "", dep))
        ok = ok and good
    print("me11_graph_same: %d graphs, %d nodes; dependencies compared in %d graphs, uncompared in %d (no base "
          "edges): %s" % (len(files), total, edges_compared, edges_uncompared,
                          ("PASS (nodes only where uncompared)" if edges_uncompared else "PASS") if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
