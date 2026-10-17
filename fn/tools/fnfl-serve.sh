#!/bin/bash
# FNFL's server (SV1): start | stop | status of llama-server on Carlos, port 8190 (web UI, OpenAI and llama APIs).
#   start [--temp T] [--top-k K] [--top-p P] [--min-p P]   (TK1: the server's sampling; defaults T 1.0, top-k 20,
#           top-p 0.95, min-p 0.05, the model's own; a request's fields override them)
#           [--spec-d N] [--draft-attention dense|sparse]   (SP2: the drafts a step proposes, 1..4, and the draft's
#           attention; defaults d 4 and sparse, the fnfl-stable server's settings; always passed as --fnfl-spec-d and
#           --fnfl-draft-attention)
#           refuses if a compute process is on either card or a canonical lock is held; takes the three canonical locks
#           (flock -n) and starts llama-server with them, so the server process itself holds the locks for its
#           lifetime (this script closes its own copies; no other process gets them); waits for /health (600 s).
#   stop    SIGTERM to the pidfile's process only, up to 60 s for it to exit, then checks both cards at 0 MiB and the
#           locks free.
#   status  the PID, /health and the lock state.
# Environment: BIN (default engine/build-fnfl/bin/llama-server), SERVE_DIR (default work/SV1/serve: server.log and
# server.pid), PORT (default 8190), CTX (default 262144: full 256K context; explicit overrides are supported).
set -u
ROOT=/home/jeremy/src/flashnext
LK=/home/jeremy/src/flashnext-opt/artifacts
BIN=${BIN:-$ROOT/engine/build-fnfl/bin/llama-server}
SERVE_DIR=${SERVE_DIR:-$ROOT/work/SV1/serve}
PORT=${PORT:-8190}
CTX=${CTX:-262144}
MODEL=/home/jeremy/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
DRAFT=/home/jeremy/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
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
SPEC_D=4
DRAFT_ATTN=sparse

case "${1:-}" in
    start)
        shift
        while [ $# -gt 0 ]; do
            [ $# -ge 2 ] || { echo "fnfl-serve: $1: no value"; exit 2; }
            case "$1" in
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
                        "[--spec-d N] [--draft-attention dense|sparse])"
                    exit 2
                    ;;
            esac
            shift 2
        done
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
        exec 7> $LK/gpu.lock 8> $LK/gpu0.lock 9> $LK/gpu1.lock
        if ! { flock -n 7 && flock -n 8 && flock -n 9; }; then
            echo "fnfl-serve: refused: a canonical lock is held"
            exit 3
        fi
        {
            echo "start=$(date -u +%FT%T) bin=$BIN sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
            echo "context: $CTX tokens (-c); port: $PORT"
            echo "sampling: temp $TEMP top-k $TOP_K top-p $TOP_P min-p $MIN_P"
            echo "spec: --fnfl-spec-d $SPEC_D --fnfl-draft-attention $DRAFT_ATTN"
        } > "$SERVE_DIR/start.txt"
        # the server inherits descriptors 7-9 and so holds the three locks; setsid detaches it from this shell
        nohup setsid "$BIN" --host 0.0.0.0 --port "$PORT" -m "$MODEL" -md "$DRAFT" -c "$CTX" -np 1 --jinja \
            --temp "$TEMP" --top-k "$TOP_K" --top-p "$TOP_P" --min-p "$MIN_P" --fnfl-draft-subset "$SUBSET" \
            --fnfl-spec-d "$SPEC_D" --fnfl-draft-attention "$DRAFT_ATTN" \
            >> "$LOG" 2>&1 < /dev/null &
        echo $! > "$PIDF"
        exec 7>&- 8>&- 9>&-
        t0=$(date +%s)
        while [ $(($(date +%s) - t0)) -lt 600 ]; do
            if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
                echo "fnfl-serve: pid $(cat "$PIDF") serving on port $PORT after $(($(date +%s) - t0)) s; log $LOG"
                exit 0
            fi
            alive || { echo "fnfl-serve: the server exited during load:"; tail -5 "$LOG"; exit 1; }
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
        ;;
    *)
        echo "usage: fnfl-serve.sh start [--temp T] [--top-k K] [--top-p P] [--min-p P]|stop|status"
        echo "       (BIN, SERVE_DIR, PORT from the environment)"
        exit 2
        ;;
esac
