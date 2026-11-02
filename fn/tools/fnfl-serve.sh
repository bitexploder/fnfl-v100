#!/bin/bash
# FNFL's server (SV1): start | stop | status of llama-server on this machine, port 8190 (web UI, OpenAI and llama APIs).
#   start [--temp T] [--top-k K] [--top-p P] [--min-p P]   (TK1: the server's sampling; defaults T 1.0, top-k 20,
#           top-p 0.95, min-p 0.05, the model's own; a request's fields override them)
#           [--spec-d N] [--draft-attention dense|sparse]   (SP2: the drafts a step proposes, 1..4, and the draft's
#           attention; defaults d 2 (operator 2026-10-26, DeepSWE) and sparse; always passed as --fnfl-spec-d and
#           --fnfl-draft-attention)
#           refuses if a compute process is on either card or a canonical lock is held; takes the three canonical locks
#           (flock -n) and starts llama-server with them, so the server process itself holds the locks for its
#           lifetime (this script closes its own copies; no other process gets them); waits for /health (600 s).
#           [--split [--split-layer K] [--slots N] [--tcp | --roce]]   (TTTS1: Flash-Next across two boxes,
#           resident TP on each: this server the leader, the front, layers 0 .. K - 1; the follower, `fnfl-drive stage`
#           (the back: layers K .. 47, the head and the draft), started on the follower over ssh, holding its three
#           canonical locks for its lifetime; N slots (default 8) in two pipelined groups.
#           PRD1, the production launch: with --split, MODEL defaults to UD-Q4_K_XL (the one-box default stays
#           IQ3_XXS), K to 26 for UD-Q4_K_XL (27 otherwise: TTTS7), CTX (the KV pool) to 1343488 for UD-Q4_K_XL (MUP2
#           piece 5: split-plan's largest q8 pool keeping 384 MiB free a card, 0.43 GiB a front card reserved for
#           VIS1's encoder), else 102,400 cells a slot; each slot's context min(CTX, 262144).
#           The link: RoCE (UCX_TLS rc_x,self, UCX_ROCE_TLS overrides, on mlx5_0:1, GID index 3, both processes on the
#           UCX built with verbs in UCX_PREFIX, default third_party/ucx-1.20.0-verbs, through LD_LIBRARY_PATH) when
#           every check passes on both boxes: the prefix's libucp is there, the login may lock unlimited memory
#           (ulimit -l; RDMA registration), the prefix's ucx_info -d lists rc_mlx5 on mlx5_0:1, and mlx5_0 port 1 is
#           ACTIVE. Otherwise TCP: UCX_TLS=tcp,self set here (never inherited), with one line naming the failed check;
#           start.txt records the transport and the check. --tcp forces TCP; --roce forces RoCE and refuses when a
#           check fails. The checks run before anything starts (both boxes connect only after loading).
#           Before any lock, card or load (TTTS3, TTTS7): each box's build stamp (BIN --version, FOLLOWER_BIN
#           --version: "fnfl: commit <sha>[-dirty]", each box its own; no device code is dumped at launch) must agree, and here llama-server's stamp must be fnfl-drive's; iommu=pt on
#           both boxes; no socket but TIME_WAIT on the split port on the follower's box; the UCX libraries each binary
#           resolves (ldd, with the transport's LD_LIBRARY_PATH) found, and the prefix's for RoCE. X-HELLO compares the
#           same strings again after the load. The follower listens with UCX_TCP_CM_REUSEADDR=y (its connection
#           manager is TCP under both transports), so a previous run's TIME_WAIT does not fail its listen.
#   stop    SIGTERM to the pidfile's process only, up to 60 s for it to exit, then checks both cards at 0 MiB and the
#           locks free. With a split: the follower ends at the leader's X-BYE; if it has not, SIGTERM to its pidfile's
#           process on the follower, then its cards and locks checked too.
#   status  the PID, /health and the lock state (with a split, the follower's too).
# Environment: BIN (default engine/build-fnfl/bin/llama-server), SERVE_DIR (default work/SV1/serve: server.log and
# server.pid), PORT (default 8190), CTX (default 262144: full 256K context; explicit overrides are supported),
# MODEL (the target's first shard, default IQ3_XXS on one box, UD-Q4_K_XL with --split; TTTS2: the same path on the
# follower, passed as stage --model), EXTRA_ARGS (more server flags, appended). --metrics is always passed (PRD1:
# /metrics for fmu_scen.py and llmmon).
# TTTS1: FOLLOWER (ssh name, default carlin), FOLLOWER_ADDR (default 10.25.0.2), SPLIT_PORT (default 13371),
# FOLLOWER_BIN (default fnfl-drive beside BIN, the same path on the follower), FOLLOWER_DIR (default SERVE_DIR, on the follower:
# follower.log and follower.pid), EOG (the follower's end-of-generation IDs, the vocabulary's: default
# 248044,248046,248063,248064,248065; X-HELLO refuses a set the server's differs from).
# VIS1 Stage B: MMPROJ (the vision projector; default none, text only): passed as --mmproj; with --split the follower
# runs with --image-planes (the front reads the images).
set -u
ROOT=$HOME/src/flashnext
LK=$HOME/src/flashnext-opt/artifacts
BIN=${BIN:-$ROOT/engine/build-fnfl/bin/llama-server}
SERVE_DIR=${SERVE_DIR:-$ROOT/work/SV1/serve}
PORT=${PORT:-8190}
CTX_SET=${CTX:+1} # an explicit CTX (TTTS1: else the split's pool is 102,400 cells a slot)
CTX=${CTX:-262144}
MODEL_SET=${MODEL:+1} # PRD1: unset means the one-box or the split's default, after the options
MODEL_IQ3=$HOME/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
MODEL_Q4=$HOME/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
MODEL=${MODEL:-$MODEL_IQ3}
DRAFT=$HOME/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
SUBSET=$ROOT/work/DR1/stageC/subset/draft-subset.ids
PIDF=$SERVE_DIR/server.pid
LOG=$SERVE_DIR/server.log

alive() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2> /dev/null; }
locks_free() { # 0 when all three locks can be taken (they are released again at once)
    (flock -n 7 && flock -n 8 && flock -n 9) 7> $LK/gpu.lock 8> $LK/gpu0.lock 9> $LK/gpu1.lock
}
cards() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; }

TEMP=1.0
TOP_K=20
TOP_P=0.95
MIN_P=0.05
SPEC_D=2
DRAFT_ATTN=sparse
SPLIT=0
SPLIT_K= # TTTS7: unset means the model's default, after the options
SLOTS=8
FOLLOWER=${FOLLOWER:-carlin}
FOLLOWER_ADDR=${FOLLOWER_ADDR:-10.25.0.2}
SPLIT_PORT=${SPLIT_PORT:-13371}
FOLLOWER_BIN=${FOLLOWER_BIN:-$(dirname "$BIN")/fnfl-drive}
FOLLOWER_DIR=${FOLLOWER_DIR:-$SERVE_DIR}
EOG=${EOG:-248044,248046,248063,248064,248065}
MMPROJ=${MMPROJ:-}
ROCE=0
UCX_PREFIX=${UCX_PREFIX:-$ROOT/third_party/ucx-1.20.0-verbs}
SPLITF=$SERVE_DIR/split # TTTS1: while a split serves: the follower's ssh name and directory (its follower.pid there)
# X-HELLO's identities, each box its own: a binary's build stamp (PRD1: generated on every build, printed by --version;
# "none" when it prints none). No device code is dumped at launch (operator, 2026-10-28: cuobjdump -sass of fnfl-drive
# took about 2 min a box); the build stamp identifies the build, and X-HELLO's digest field carries "none".
ID_FNS='commit_of() { local c; c=$("$1" --version 2> /dev/null | sed -n "s/^fnfl: commit //p" | head -1); echo "${c:-none}"; }'
eval "$ID_FNS"
remote() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$FOLLOWER" "$@"; }
follower_alive() { remote "[ -f $FOLLOWER_DIR/follower.pid ] && kill -0 \$(cat $FOLLOWER_DIR/follower.pid) 2> /dev/null"; }
# the follower's end: it leaves at the leader's X-BYE (or when the link is lost); else SIGTERM to its pidfile's
# process only, up to 60 s; then the follower's cards and locks
follower_stop() {
    for _ in $(seq 30); do
        follower_alive || break
        sleep 1
    done
    if follower_alive; then
        remote "kill -TERM \$(cat $FOLLOWER_DIR/follower.pid)"
        for _ in $(seq 60); do
            follower_alive || break
            sleep 1
        done
    fi
    if follower_alive; then
        echo "fnfl-serve: the follower on $FOLLOWER did not exit in 60 s"
        return 1
    fi
    rm -f "$SPLITF"
    echo "fnfl-serve: the follower on $FOLLOWER stopped; there: $(remote 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); echo "compute processes: ${a:-none}"')"
}
[ -f "$SPLITF" ] && read -r FOLLOWER FOLLOWER_DIR < "$SPLITF"
# PRD1: RoCE's checks on both boxes, before anything starts: prints the first that fails (nothing: all passed)
roce_check() {
    local box ok
    local pfx="[ -f $UCX_PREFIX/lib/libucp.so.0 ]"
    local mem='[ "$(ulimit -l)" = unlimited ]'
    local rc="LD_LIBRARY_PATH=$UCX_PREFIX/lib $UCX_PREFIX/bin/ucx_info -d 2> /dev/null | awk '/Transport:/ { t = \$3 } /Device:/ { if (t == \"rc_mlx5\" && \$3 == \"mlx5_0:1\") f = 1 } END { exit !f }'"
    local port='grep -q ACTIVE /sys/class/infiniband/mlx5_0/ports/1/state 2> /dev/null'
    for box in here "$FOLLOWER"; do
        run() { if [ "$box" = here ]; then bash -c "$1"; else remote "$1"; fi; }
        run "$pfx" || { echo "$box: no $UCX_PREFIX/lib/libucp.so.0"; return; }
        run "$mem" || { echo "$box: the login may lock $(run 'ulimit -l') KiB (ulimit -l), not unlimited: start from a fresh login"; return; }
        run "$rc" || { echo "$box: ucx_info -d lists no rc_mlx5 on mlx5_0:1"; return; }
        run "$port" || { echo "$box: mlx5_0 port 1 is not ACTIVE"; return; }
    done
}
# PRD1 (TTTS7): the UCX libraries binaries resolve (ldd), with LD_LIBRARY_PATH as the transport sets it ("" for TCP):
# "libucp.so.0=<path>" a library, "not found" kept
ucx_libs() {
    local lp=$1
    shift
    env $lp ldd "$@" 2> /dev/null | grep -E 'libuc[pst]\.so' | awk '{ print $1 "=" ($3 == "not" ? "not found" : $3) }' | sort -u | tr '\n' ' '
}

case "${1:-}" in
    start)
        shift
        TRANSPORT=auto # PRD1: auto (RoCE when its checks pass), tcp (--tcp) or roce (--roce: refused when one fails)
        while [ $# -gt 0 ]; do
            if [ "$1" = --split ]; then
                SPLIT=1
                shift
                continue
            fi
            if [ "$1" = --roce ] || [ "$1" = --tcp ]; then
                [ "$TRANSPORT" = auto ] || { echo "fnfl-serve: one of --tcp and --roce"; exit 2; }
                TRANSPORT=${1#--}
                shift
                continue
            fi
            [ $# -ge 2 ] || { echo "fnfl-serve: $1: no value"; exit 2; }
            case "$1" in
                --split-layer)
                    [[ "$2" =~ ^[0-9]+$ ]] && [ "$2" -ge 2 ] && [ "$2" -le 47 ] || {
                        echo "fnfl-serve: --split-layer is 2..47"
                        exit 2
                    }
                    SPLIT_K=$2
                    ;;
                --slots)
                    [[ "$2" =~ ^[1-8]$ ]] || { echo "fnfl-serve: --slots is 1..8"; exit 2; }
                    SLOTS=$2
                    ;;
                --temp) TEMP=$2 ;;
                --top-k) TOP_K=$2 ;;
                --top-p) TOP_P=$2 ;;
                --min-p) MIN_P=$2 ;;
                --spec-d)
                    [[ "$2" =~ ^[1-4]$ ]] || { echo "fnfl-serve: --spec-d is 1..4"; exit 2; }
                    SPEC_D=$2
                    ;;
                --draft-attention)
                    [[ "$2" =~ ^(dense|sparse)$ ]] || {
                        echo "fnfl-serve: --draft-attention is dense or sparse"
                        exit 2
                    }
                    DRAFT_ATTN=$2
                    ;;
                *)
                    echo "fnfl-serve: unknown option $1 (start [--temp T] [--top-k K] [--top-p P] [--min-p P]" \
                        "[--spec-d N] [--draft-attention dense|sparse] [--split [--split-layer K] [--slots N]" \
                        "[--tcp | --roce]])"
                    exit 2
                    ;;
            esac
            shift 2
        done
        if [ "$SPLIT" = 1 ] && [ -z "$MODEL_SET" ]; then
            MODEL=$MODEL_Q4 # PRD1: the production file on the split
        fi
        if [ -z "$SPLIT_K" ]; then
            # TTTS7 (Amendment 1, TTTS5's default): UD-Q4_K_XL splits at 26 ("Choice": the boxes' stages within one
            # layer of each other there); every other model (IQ3_XXS) at 27, as before
            case "$MODEL" in
                *UD-Q4_K_XL*) SPLIT_K=26 ;;
                *) SPLIT_K=27 ;;
            esac
        fi
        [ "$SPLIT" = 1 ] || [ "$TRANSPORT" = auto ] || { echo "fnfl-serve: --$TRANSPORT needs --split"; exit 2; }
        mkdir -p "$SERVE_DIR"
        if alive; then
            echo "fnfl-serve: already running: pid $(cat "$PIDF")"
            exit 1
        fi
        apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)
        if [ -n "$apps" ]; then
            echo "fnfl-serve: refused: a compute process is on a card: $apps"
            exit 4
        fi
        [ -x "$BIN" ] || { echo "fnfl-serve: no binary $BIN"; exit 2; }
        echo "fnfl-serve: model $MODEL"
        NP=1
        SPLIT_ARGS=()
        ROCE_WHY=
        if [ "$SPLIT" = 1 ]; then
            # ---- PRD1: every split check before anything starts (no lock, no card, no load on either box) ----
            [ -x "$FOLLOWER_BIN" ] || { echo "fnfl-serve: refused: no follower binary $FOLLOWER_BIN here (its build is the leader's)"; exit 2; }
            # the transport: RoCE's four checks on both boxes, unless --tcp
            if [ "$TRANSPORT" = tcp ]; then
                ROCE_WHY="--tcp"
            else
                ROCE_WHY=$(roce_check)
            fi
            if [ -z "$ROCE_WHY" ]; then
                ROCE=1
            elif [ "$TRANSPORT" = roce ]; then
                echo "fnfl-serve: refused: --roce: $ROCE_WHY"
                exit 2
            else
                echo "fnfl-serve: RoCE not used ($ROCE_WHY): the link over TCP, UCX_TLS=tcp,self"
            fi
            # the builds: each box's stamp and device code, here llama-server's stamp fnfl-drive's
            COMMIT=$(commit_of "$BIN")
            HCOMMIT=$(commit_of "$FOLLOWER_BIN")
            DIGEST=none
            FDIGEST=none
            FCOMMIT=$(remote "$ID_FNS; if [ -x $FOLLOWER_BIN ]; then commit_of $FOLLOWER_BIN; else echo missing; fi")
            if [ "$COMMIT" = none ] || [ "$COMMIT" != "$HCOMMIT" ]; then
                echo "fnfl-serve: refused: llama-server's build stamp ($COMMIT) is not this box's fnfl-drive's ($HCOMMIT)"
                exit 3
            fi
            if [ "$FCOMMIT" != "$COMMIT" ]; then
                echo "fnfl-serve: refused: the builds differ: here commit $COMMIT; on $FOLLOWER ($FOLLOWER_BIN) commit ${FCOMMIT:-no answer}"
                exit 3
            fi
            # TTTS7's checks (work/TTTS7/bin/preload-check.sh): iommu=pt, the split port, the UCX each binary resolves
            [ "$(grep -c iommu=pt /proc/cmdline)" -ge 1 ] || { echo "fnfl-serve: refused: no iommu=pt in this box's /proc/cmdline"; exit 2; }
            [ "$(remote 'grep -c iommu=pt /proc/cmdline')" -ge 1 ] 2> /dev/null || { echo "fnfl-serve: refused: no iommu=pt in $FOLLOWER's /proc/cmdline"; exit 2; }
            socks=$(remote "ss -Htan '( sport = :$SPLIT_PORT )' | awk '\$1 != \"TIME-WAIT\"' | wc -l")
            [ "$socks" = 0 ] || { echo "fnfl-serve: refused: $socks socket(s) other than TIME_WAIT on port $SPLIT_PORT on $FOLLOWER (another split?)"; exit 3; }
            if [ "$ROCE" = 1 ]; then LP="LD_LIBRARY_PATH=$UCX_PREFIX/lib"; else LP=; fi
            IMPL=$(dirname "$BIN")/libllama-server-impl.so
            ucx_here=$(ucx_libs "$LP" "$BIN" $([ -f "$IMPL" ] && echo "$IMPL"))
            ucx_there=$(remote "$(declare -f ucx_libs); ucx_libs '$LP' $FOLLOWER_BIN")
            for u in "here:$ucx_here" "$FOLLOWER:$ucx_there"; do
                if grep -q "not found" <<< "${u#*:}" || [ "$(grep -o 'libuc[pst]\.so\.0' <<< "${u#*:}" | sort -u | wc -l)" -lt 3 ]; then
                    echo "fnfl-serve: refused: UCX on ${u%%:*}: ${u#*:}"
                    exit 2
                fi
                if [ "$ROCE" = 1 ] && grep -E 'libuc[pst]\.so\.0=' <<< "${u#*:}" | grep -qv "=$UCX_PREFIX/lib/"; then
                    echo "fnfl-serve: refused: RoCE: a binary on ${u%%:*} resolves UCX outside $UCX_PREFIX: ${u#*:}"
                    exit 2
                fi
            done
            XID="front $(journalctl -k -b 2> /dev/null | grep -c 'NVRM: Xid') $FOLLOWER $(remote "journalctl -k -b 2> /dev/null | grep -c 'NVRM: Xid'")"
        fi
        exec 7> $LK/gpu.lock 8> $LK/gpu0.lock 9> $LK/gpu1.lock
        if ! { flock -n 7 && flock -n 8 && flock -n 9; }; then
            echo "fnfl-serve: refused: a canonical lock is held"
            exit 3
        fi
        VIS_ARGS=()
        [ -n "$MMPROJ" ] && VIS_ARGS=(--mmproj "$MMPROJ")
        if [ "$SPLIT" = 1 ]; then
            # TTTS1: the follower first, under its three canonical locks (held by the follower itself)
            NP=$SLOTS
            if [ -z "$CTX_SET" ]; then
                # MUP2 piece 5: fnfl-drive split-plan --k 26 --slots 8 --slot-cells 262144 --cal (MUP1 Load 3's
                # server: pool 819,200, free 3,177 / 3,177 MiB on the front, 3,423 / 3,411 MiB on the back) --floor-mib 384
                # --reserve-front 0.43 (work/MUP2/runs/plan/split-plan.txt)
                case "$MODEL" in
                    *UD-Q4_K_XL*) CTX=1343488 ;;
                    *) CTX=$((102400 * SLOTS)) ;;
                esac
            fi
            CELLS=$((CTX < 262144 ? CTX : 262144))
            if [ "$ROCE" = 1 ]; then
                # TTTS7: the link over RoCE, both processes on the prefix's UCX (the same path on both boxes)
                export UCX_TLS=${UCX_ROCE_TLS:-rc_x,self} UCX_NET_DEVICES=mlx5_0:1 UCX_IB_GID_INDEX=${UCX_IB_GID_INDEX:-3}
                export LD_LIBRARY_PATH=$UCX_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
                FENV="LD_LIBRARY_PATH=$UCX_PREFIX/lib UCX_TLS=$UCX_TLS UCX_NET_DEVICES=mlx5_0:1 UCX_IB_GID_INDEX=$UCX_IB_GID_INDEX"
            else
                # PRD1: TCP is set here, never inherited
                export UCX_TLS=tcp,self UCX_NET_DEVICES=${UCX_NET_DEVICES:-enp3s0np0} UCX_IB_GID_INDEX=${UCX_IB_GID_INDEX:-3}
                FENV="UCX_TLS=$UCX_TLS UCX_NET_DEVICES=${FOLLOWER_NET:-enp5s0np0} UCX_IB_GID_INDEX=$UCX_IB_GID_INDEX"
            fi
            # PRD1 (TTTS3): the follower listens through UCX's TCP connection manager under both transports; reuse lets
            # it bind beside a previous run's TIME_WAIT
            FENV="$FENV UCX_TCP_CM_REUSEADDR=y"
            FARGS="stage --kv-format q8 --model $MODEL --split-layer $SPLIT_K --draft-attention $DRAFT_ATTN --draft-subset $SUBSET --slots $SLOTS --cells $CELLS --pool-cells $CTX --spec-d $SPEC_D --eog $EOG --split-addr $FOLLOWER_ADDR --split-port $SPLIT_PORT"
            [ -n "$MMPROJ" ] && FARGS="$FARGS --image-planes"
            # the follower with the identities compared above (X-HELLO compares them with the leader's)
            fstart="mkdir -p $FOLLOWER_DIR; apps=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); [ -z \"\$apps\" ] || { echo \"BUSY \$apps\"; exit 4; }; exec 7> $LK/gpu.lock 8> $LK/gpu0.lock 9> $LK/gpu1.lock; { flock -n 7 && flock -n 8 && flock -n 9; } || { echo LOCKED; exit 3; }; env $FENV nohup setsid $FOLLOWER_BIN $FARGS --split-commit $FCOMMIT --split-digest $FDIGEST >> $FOLLOWER_DIR/follower.log 2>&1 < /dev/null & echo \$! > $FOLLOWER_DIR/follower.pid; echo STARTED \$!"
            fout=$(remote "$fstart")
            case "$fout" in
                STARTED*) ;;
                *)
                    echo "fnfl-serve: refused: the follower on $FOLLOWER did not start: ${fout:-no answer}"
                    exit 3
                    ;;
            esac
            SPLIT_ARGS=(--fnfl-split-layer "$SPLIT_K" --fnfl-split-addr "$FOLLOWER_ADDR" --fnfl-split-port "$SPLIT_PORT"
                --fnfl-split-commit "$COMMIT" --fnfl-split-digest "$DIGEST" --fnfl-kv-format q8)
            echo "$FOLLOWER $FOLLOWER_DIR" > "$SPLITF"
            read -r _ FPID <<< "$fout"
            echo "fnfl-serve: split at k $SPLIT_K: the follower on $FOLLOWER ($FOLLOWER_ADDR:$SPLIT_PORT), pid $FPID, log $FOLLOWER_DIR/follower.log"
            echo "fnfl-serve: X-HELLO: here commit $COMMIT, device code $DIGEST; there commit $FCOMMIT, device code $FDIGEST"
            if [ "$ROCE" = 1 ]; then
                echo "fnfl-serve: transport RoCE: UCX_TLS=$UCX_TLS UCX_NET_DEVICES=$UCX_NET_DEVICES UCX_IB_GID_INDEX=$UCX_IB_GID_INDEX, UCX from $UCX_PREFIX on both boxes"
            else
                echo "fnfl-serve: transport TCP ($ROCE_WHY): UCX_TLS=$UCX_TLS UCX_NET_DEVICES=$UCX_NET_DEVICES (follower ${FOLLOWER_NET:-enp5s0np0}) UCX_IB_GID_INDEX=$UCX_IB_GID_INDEX"
            fi
            echo "fnfl-serve: $SLOTS slots, KV pool $CTX cells, each slot's context $CELLS, --fnfl-spec-d $SPEC_D, eog $EOG"
        fi
        {
            echo "start=$(date -u +%FT%T) bin=$BIN sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
            echo "model: $MODEL"
            echo "context: $CTX tokens (-c); port: $PORT"
            echo "sampling: temp $TEMP top-k $TOP_K top-p $TOP_P min-p $MIN_P"
            echo "spec: --fnfl-spec-d $SPEC_D --fnfl-draft-attention $DRAFT_ATTN"
            echo "metrics: --metrics"
            if [ "$SPLIT" = 1 ]; then
                echo "split: k $SPLIT_K, $SLOTS slots, KV pool $CTX cells, each slot's context $CELLS, follower $FOLLOWER ($FOLLOWER_ADDR:$SPLIT_PORT) $FOLLOWER_BIN, commit $COMMIT, device code $DIGEST, UCX_TLS=$UCX_TLS"
                if [ "$ROCE" = 1 ]; then
                    echo "transport: RoCE (UCX from $UCX_PREFIX); every RoCE check passed on both boxes"
                else
                    echo "transport: TCP (UCX_TLS=tcp,self); RoCE not used: $ROCE_WHY"
                fi
                echo "xid before: $XID"
            fi
            [ -n "${EXTRA_ARGS:-}" ] && echo "extra: $EXTRA_ARGS"
            [ -n "$MMPROJ" ] && echo "vision: --mmproj $MMPROJ$([ "$SPLIT" = 1 ] && echo ", follower --image-planes")"
        } > "$SERVE_DIR/start.txt"
        # the server inherits descriptors 7-9 and so holds the three locks; setsid detaches it from this shell
        nohup setsid "$BIN" --host 0.0.0.0 --port "$PORT" -m "$MODEL" -md "$DRAFT" -c "$CTX" -np "$NP" --jinja \
            --temp "$TEMP" --top-k "$TOP_K" --top-p "$TOP_P" --min-p "$MIN_P" --fnfl-draft-subset "$SUBSET" \
            --fnfl-spec-d "$SPEC_D" --fnfl-draft-attention "$DRAFT_ATTN" --metrics "${SPLIT_ARGS[@]}" "${VIS_ARGS[@]}" ${EXTRA_ARGS:-} \
            >> "$LOG" 2>&1 < /dev/null &
        echo $! > "$PIDF"
        exec 7>&- 8>&- 9>&-
        t0=$(date +%s)
        while [ $(($(date +%s) - t0)) -lt 600 ]; do
            if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
                echo "fnfl-serve: pid $(cat "$PIDF") serving on port $PORT after $(($(date +%s) - t0)) s; log $LOG"
                exit 0
            fi
            if ! alive; then
                echo "fnfl-serve: the server exited during load:"
                tail -5 "$LOG"
                [ "$SPLIT" = 1 ] && follower_stop
                exit 1
            fi
            sleep 1
        done
        echo "fnfl-serve: no /health in 600 s (pid $(cat "$PIDF") still loading?)"
        exit 1
        ;;
    stop)
        if ! alive; then
            echo "fnfl-serve: not running"
        else
            pid=$(cat "$PIDF")
            kill -TERM "$pid"
            for _ in $(seq 60); do
                kill -0 "$pid" 2> /dev/null || break
                sleep 1
            done
            if kill -0 "$pid" 2> /dev/null; then
                echo "fnfl-serve: pid $pid did not exit in 60 s"
                exit 1
            fi
            echo "fnfl-serve: pid $pid stopped"
        fi
        if [ -f "$SPLITF" ]; then
            follower_stop || exit 1
        fi
        sleep 1
        apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)
        echo "fnfl-serve: cards: $(cards); compute processes: ${apps:-none}; locks: $(locks_free && echo free || echo HELD)"
        [ -z "$apps" ] && locks_free && ! cards | grep -qE ', [1-9][0-9]* MiB'
        ;;
    status)
        if alive; then
            echo "fnfl-serve: pid $(cat "$PIDF") running"
        else
            echo "fnfl-serve: not running"
        fi
        echo "fnfl-serve: /health: $(curl -s -m 5 "http://127.0.0.1:$PORT/health" || echo unreachable)"
        if locks_free; then
            echo "fnfl-serve: locks: free"
        else
            echo "fnfl-serve: locks: held by $(lslocks -n -o PID,PATH 2> /dev/null | grep -E 'gpu[01]?\.lock' | awk '{print $1}' | sort -u | tr '\n' ' ')"
        fi
        echo "fnfl-serve: cards: $(cards)"
        if [ -f "$SPLITF" ]; then
            if follower_alive; then
                echo "fnfl-serve: the follower on $FOLLOWER: pid $(remote "cat $FOLLOWER_DIR/follower.pid") running"
            else
                echo "fnfl-serve: the follower on $FOLLOWER: not running"
            fi
        fi
        ;;
    *)
        echo "usage: fnfl-serve.sh start [--temp T] [--top-k K] [--top-p P] [--min-p P] [--spec-d N]" \
            "[--draft-attention dense|sparse] [--split [--split-layer K] [--slots N] [--tcp | --roce]]|stop|status"
        echo "       (BIN, SERVE_DIR, PORT from the environment)"
        exit 2
        ;;
esac
