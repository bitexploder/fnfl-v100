#!/usr/bin/env bash
# ME10: the SASS-unchanged row (request 15's R15-07, repaired). For each object, dump BASE's and CAND's SASS, compare
# with me10_sass_same.py and classify any change with me10_sass_classify.py, keep the small reports, and delete the large
# dumps once compared (about 1.1 GB a run). The status is aggregated over every object: any failed comparison, missing
# object or dump failure makes the row fail, even when later objects pass and the cleanup succeeds.
#
# usage: me10_sass_check.sh BASE_BUILD CAND_BUILD OUT_DIR TOOLS_DIR OBJECT...
#   e.g. me10_sass_check.sh build-base build out/sass engine/fn/tools experts_gu_iq2 experts_gu_iq3 experts_cache
set -u
base=$1 cand=$2 out=$3 tools=$4
shift 4
cuobjdump=${CUOBJDUMP:-/usr/local/cuda-12.8/bin/cuobjdump}
mkdir -p "$out" || exit 2
status=0
summary="$out/sass-check-summary.txt"
: > "$summary"
for f in "$@"; do
    bo=$(find "$base" -name "$f.cu.o" -print -quit)
    co=$(find "$cand" -name "$f.cu.o" -print -quit)
    if [ -z "$bo" ] || [ -z "$co" ]; then
        echo "$f: object missing (base '$bo', candidate '$co'): FAIL" | tee -a "$summary"
        status=1
        continue
    fi
    b="$out/$f.before.sass" a="$out/$f.after.sass"
    if ! "$cuobjdump" -sass "$bo" > "$b" || ! "$cuobjdump" -sass "$co" > "$a" ||
        ! "$cuobjdump" -res-usage "$bo" > "$out/$f.before.res" || ! "$cuobjdump" -res-usage "$co" > "$out/$f.after.res"; then
        echo "$f: cuobjdump failed: FAIL" | tee -a "$summary"
        status=1
    else
        python3 -I "$tools/me10_sass_same.py" "$b" "$a" > "$out/$f.same.txt"
        rc=$?
        python3 -I "$tools/me10_sass_classify.py" "$b" "$a" "$out/$f.before.res" "$out/$f.after.res" > "$out/$f.classify.txt"
        echo "$f: $(tail -1 "$out/$f.same.txt"); $(tail -1 "$out/$f.classify.txt") (rc $rc)" | tee -a "$summary"
        [ "$rc" -eq 0 ] || status=1
    fi
    rm -f "$b" "$a" # the dumps are large; the reports above and the .res files stay
done
echo "me10_sass_check: $( [ $status -eq 0 ] && echo PASS || echo FAIL ) (every object's status aggregated)" | tee -a "$summary"
exit $status
