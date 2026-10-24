#!/usr/bin/env bash
# Small CPU/MPS LLWM run. Planning CORE can still be expensive on a laptop.
set -euo pipefail
export NANOCHAT_DEVICE="${NANOCHAT_DEVICE:-cpu}"
export DEPTH="${DEPTH:-4}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-128}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-200}"
export DATA_SHARDS="${DATA_SHARDS:-8}"
export EVAL_TOKENS="${EVAL_TOKENS:-1024}"
export EVAL_EVERY="${EVAL_EVERY:-100}"
export SAMPLE_EVERY="${SAMPLE_EVERY:-100}"
export CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-200}"
export PLAN_ITERATIONS="${PLAN_ITERATIONS:-5}"
export SFT_NUM_ITERATIONS="${SFT_NUM_ITERATIONS:-100}"
export CHAT_MAX_NEW_TOKENS="${CHAT_MAX_NEW_TOKENS:-32}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/speedrun.sh"
