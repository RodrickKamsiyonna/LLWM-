#!/usr/bin/env bash
# Run on a TPU VM host to inspect one persistent LLWM run.
set -euo pipefail
: "${RUN_ID:?Set RUN_ID}"
: "${ATTEMPT_ID:?Set ATTEMPT_ID}"
root="${CHECKPOINT_ROOT:-/srv/llwm-checkpoints}"
repo="${LLWM_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
session="llwm-${ATTEMPT_ID}"
echo "=== $(hostname) | run=$RUN_ID attempt=$ATTEMPT_ID ==="
if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux: running ($session)"
    tmux display-message -p -t "$session" 'pane_pid=#{pane_pid} pane_current_command=#{pane_current_command}'
else
    echo "tmux: no session ($session)"
fi
echo "=== checkpoint manifests ==="
find "$root/$RUN_ID" -maxdepth 1 -name 'complete_*.json' -type f -print 2>/dev/null | sort -V | tail -3 || true
echo "=== host log tail ==="
tail -n 30 "$repo/evidence/${ATTEMPT_ID}-host.log" 2>/dev/null || true
echo "=== rank log tails ==="
rank_dir="$repo/logs/$RUN_ID/$ATTEMPT_ID"
if [[ -d "$rank_dir" ]]; then
    find "$rank_dir" -type f \( -name '*.stdout.log' -o -name '*.jsonl' -o -name '*.fault.log' \) -print | sort | while read -r file; do
        echo "--- $file"
        tail -n 5 "$file"
    done
else
    echo "No per-rank log directory yet: $rank_dir"
fi
