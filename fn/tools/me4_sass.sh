#!/bin/bash
# ME4 (expires at ME4's disposition): the prompt expert kernels' ptxas resources and SASS, this tree against BASE (a
# commit), on the CPU (nvcc and cuobjdump; no device, no model). Compiles fn/kernels/experts/experts_prompt.cu at BASE
# and here with the same flags (sm_70, the fn/ targets' math flags, -Xptxas -v), dumps both, and compares
# (me4_sass.py): every resident instantiation (X = 0), ep_group and ep_sum identical, and the cache instantiations'
# (X = 1) spills no greater than BASE's.
# usage (from the engine root): fn/tools/me4_sass.sh BASE OUT_DIR
set -euo pipefail
base=$1
out=$2
nv=/usr/local/cuda-12.8/bin
mkdir -p "$out/base"
git archive "$base" fn | tar -x -C "$out/base"
flags=(-arch=sm_70 -O3 -lineinfo -ccbin g++-14 -std=c++17 --ftz=false --prec-div=true --prec-sqrt=true --fmad=true
       -Xptxas -v)
"$nv/nvcc" "${flags[@]}" -I"$out/base" -c "$out/base/fn/kernels/experts/experts_prompt.cu" -o "$out/base.o" \
    2> "$out/ptxas-base.txt"
"$nv/nvcc" "${flags[@]}" -I. -c fn/kernels/experts/experts_prompt.cu -o "$out/me4.o" 2> "$out/ptxas-me4.txt"
"$nv/cuobjdump" -sass "$out/base.o" > "$out/base.sass"
"$nv/cuobjdump" -sass "$out/me4.o" > "$out/me4.sass"
python3 -I fn/tools/me4_sass.py "$out"
