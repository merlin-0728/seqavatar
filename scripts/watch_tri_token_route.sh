#!/bin/bash
set -euo pipefail

GPU2_SESSION=${GPU2_SESSION:-tri_token_route_gpu2}
GPU3_SESSION=${GPU3_SESSION:-tri_token_route_gpu3}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-600}
LOG_FILE=${LOG_FILE:-/media/coding/ckx/human/SeqAvatar/logs/tri/tri_token_route_watch.log}

mkdir -p "$(dirname "$LOG_FILE")"

while true; do
    {
        echo "=== $(date +%F\ %T) ==="
        if tmux has-session -t "$GPU2_SESSION" 2>/dev/null || tmux has-session -t "$GPU3_SESSION" 2>/dev/null; then
            echo "tri_token_route still running"
        else
            echo "tri_token_route finished; ready for next step"
            exit 0
        fi
    } >> "$LOG_FILE"
    sleep "$INTERVAL_SECONDS"
done
