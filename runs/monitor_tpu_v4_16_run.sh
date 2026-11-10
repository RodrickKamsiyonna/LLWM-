#!/usr/bin/env bash
# Run locally on the Mac to follow this TPU run's losses and planned samples.
set -euo pipefail

TPU_NAME="${TPU_NAME:-node-1}"
ZONE="${ZONE:-us-central2-b}"
PROJECT="${PROJECT:-llwm-512507}"
RUN_ID="${RUN_ID:?Set RUN_ID}"
ATTEMPT_ID="${ATTEMPT_ID:?Set ATTEMPT_ID}"
REMOTE_REPO="${REMOTE_REPO:-/home/macbookpro/LLWM_resume12k}"
WORKER="${WORKER:-all}"
INITIAL_LINES="${INITIAL_LINES:-200}"

for value in "$TPU_NAME" "$ZONE" "$PROJECT" "$RUN_ID" "$ATTEMPT_ID"; do
    if [[ "$value" =~ [^A-Za-z0-9._-] ]]; then
        echo "Unexpected character in monitor setting: $value" >&2
        exit 2
    fi
done
[[ "$WORKER" == all || "$WORKER" =~ ^[0-9]+$ ]] || { echo 'WORKER must be all or an index' >&2; exit 2; }
[[ "$INITIAL_LINES" =~ ^[0-9]+$ ]] || { echo 'INITIAL_LINES must be an integer' >&2; exit 2; }
[[ "$REMOTE_REPO" =~ ^/[A-Za-z0-9_./-]+$ ]] || { echo 'Unexpected remote repository path' >&2; exit 2; }

remote_dir="$REMOTE_REPO/logs/$RUN_ID/$ATTEMPT_ID"
# Keep full output: filtering to loss/sample lines hid the actual traceback.
# Read each host's own files; rank 0 can be placed on either TPU VM.
remote_script="shopt -s nullglob; files=($remote_dir/*/*bootstrap.stdout.log); if (( \${#files[@]} == 0 )); then echo 'No rank stdout logs found: $remote_dir' >&2; exit 1; fi; tail -n $INITIAL_LINES -F \"\${files[@]}\""
printf -v remote_command 'sudo -u macbookpro bash -c %q' "$remote_script"
exec gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" --worker="$WORKER" --batch-size=all --command="$remote_command"
