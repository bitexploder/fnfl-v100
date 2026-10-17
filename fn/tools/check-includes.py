#!/usr/bin/env python3
"""The hub-header rule (FL2) and the no-borrowed-kernel rule (C1). Fails, listing each violation, when:

- a file under fn/ outside fn/borrowed/ includes llama.h, ggml.h, ggml-impl.h, ggml-backend*.h, gguf.h (fn/glue/ may)
  or common/common.h;
- any file under fn/, fn/borrowed/ included, includes a ggml-cuda header (ggml-cuda/..., ggml-cuda.h, or a header
  named as one in ggml/src/ggml-cuda/);
- any file under fn/, fn/borrowed/ included, includes a ggml-base quant header (ggml-quants.h, ggml-common.h): FNFL's
  decoders are fn/quant/ (C1b);
- any file under fn/ names an fnfl_borrowed_ symbol (the prefix of the GPU adapters C1 removed);
- fn/CMakeLists.txt names ggml-cuda outside a comment (no fn/ target links it);
- fn/fnfl.h includes a CUDA header;
- a file outside fn/ includes an fn/ header other than fn/fnfl.h.

usage: check-includes.py ENGINE_ROOT [STAMP]   (writes STAMP on success)
"""
import os
import re
import sys

SOURCE_EXT = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".inc")
SKIP_DIRS = {".git", "build", "node_modules"}
INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)
HUB = {"llama.h", "ggml.h", "ggml-impl.h", "gguf.h"}
QUANT = {"ggml-quants.h", "ggml-common.h"}
BORROWED_SYMBOL = re.compile(r"\bfnfl_borrowed_\w*")
CUDA = re.compile(r"(^|/)(cuda[^/]*\.h|cublas[^/]*\.h|cub/|thrust/|nvtx[^/]*|cooperative_groups|mma\.h|[^/]*\.cuh)")


def sources(root):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith("build")]
        for f in files:
            if f.endswith(SOURCE_EXT):
                yield os.path.join(d, f)


def includes(path):
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return []
    out = []
    for m in INCLUDE.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        out.append((line, m.group(2)))
    return out


def symbols(path):
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return []
    return [(text.count("\n", 0, m.start()) + 1, m.group(0)) for m in BORROWED_SYMBOL.finditer(text)]


def cmake_lines(path):
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    except OSError:
        return []
    return [(i + 1, l.split("#", 1)[0]) for i, l in enumerate(lines)]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = os.path.realpath(sys.argv[1])
    fn_dir = os.path.join(root, "fn")
    fnfl_h = os.path.join(fn_dir, "fnfl.h")
    cuda_hdrs = set()
    for d, _, files in os.walk(os.path.join(root, "ggml", "src", "ggml-cuda")):
        cuda_hdrs.update(f for f in files if f.endswith((".h", ".cuh")))

    bad = []
    for path in sources(root):
        rel = os.path.relpath(path, root)
        in_fn = rel.startswith("fn" + os.sep)
        in_borrowed = rel.startswith(os.path.join("fn", "borrowed") + os.sep)
        in_glue = rel.startswith(os.path.join("fn", "glue") + os.sep)
        for line, inc in includes(path):
            base = os.path.basename(inc)
            where = f"{rel}:{line}: #include \"{inc}\""
            if in_fn and not inc.startswith("fn/") and ("ggml-cuda" in inc or base in cuda_hdrs):
                bad.append(f"{where}: ggml-cuda header in fn/ (no borrowed kernels)")
            elif in_fn and not inc.startswith("fn/") and base in QUANT:
                bad.append(f"{where}: ggml-base quant header in fn/ (FNFL's decoders are fn/quant/)")
            elif in_fn and not in_borrowed and not inc.startswith("fn/"):
                hub = (base in HUB and not (base == "gguf.h" and in_glue)) or base.startswith("ggml-backend") \
                    or base == "common.h"
                if hub:
                    bad.append(f"{where}: borrowed hub header outside fn/borrowed/")
            if os.path.realpath(path) == fnfl_h and CUDA.search(inc):
                bad.append(f"{where}: fn/fnfl.h must not include a CUDA header")
            if not in_fn:
                target = None
                if inc.startswith("fn/"):
                    target = os.path.join(root, inc)
                elif "fn/" in inc:
                    target = os.path.realpath(os.path.join(os.path.dirname(path), inc))
                if target is not None and os.path.realpath(target).startswith(fn_dir + os.sep) \
                        and os.path.realpath(target) != fnfl_h:
                    bad.append(f"{where}: only fn/fnfl.h may be included from outside fn/")

        if in_fn:
            for line, sym in symbols(path):
                bad.append(f"{rel}:{line}: {sym}: a borrowed GPU adapter symbol (no borrowed kernels)")

    for line, text in cmake_lines(os.path.join(fn_dir, "CMakeLists.txt")):
        if "ggml-cuda" in text:
            bad.append(f"fn/CMakeLists.txt:{line}: names ggml-cuda (no fn/ target links it)")

    if bad:
        print("check-includes: %d violation(s) of the hub-header or no-borrowed-kernel rule (fnfl/STANDARDS.md)"
              % len(bad), file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        sys.exit(1)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write("ok\n")


if __name__ == "__main__":
    main()
