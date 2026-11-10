#!/usr/bin/env bash
# Launch on both TPU hosts with the same RUN_ID and ATTEMPT_ID.
set -euo pipefail
: "${RUN_ID:?Set a unique RUN_ID shared by both hosts}"
: "${CHECKPOINT_ROOT:?Set the shared server checkpoint root}"
: "${DATASET_MANIFEST:?Set this host successful full preflight manifest}"
export ATTEMPT_ID="${ATTEMPT_ID:-${RUN_ID}-a1}"
export DEPTH="${DEPTH:-15}" PREDICTOR_LAYERS="${PREDICTOR_LAYERS:-6}"
export ASPECT_RATIO="${ASPECT_RATIO:-110}" HEAD_DIM="${HEAD_DIM:-128}" WINDOW_PATTERN=L
export ACTION_DIM=128 ACTION_LAYERS="${ACTION_LAYERS:-9}"
export ACTION_CONDITIONING=next_token HEAD_SOURCE=predictor HEAD_TRAINING=joint
export PREDICTION_HORIZON=1 MEMORY_WEIGHT=0 CE_WEIGHT=1 CE_THRESHOLD=0 CE_POWER=1
export KL_MODE=target TARGET_KL="${TARGET_KL:-1}" KL_LAMBDA="${KL_LAMBDA:-3}"
export PLANNING_MODE=straight_through PLANNING_EVERY="${PLANNING_EVERY:-500}"
export PLAN_ITERATIONS="${PLAN_ITERATIONS:-100}" PLAN_LR="${PLAN_LR:-0.01}"
export PLAN_EXECUTE_STEPS="${PLAN_EXECUTE_STEPS:-10}" EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-20}"
export PLAN_GOAL="${PLAN_GOAL:-.}"
export PLANNING_PROMPT="${PLANNING_PROMPT:-What is the tallest mountain?}"
export PLANNING_PROMPT_2="${PLANNING_PROMPT_2:-The traveler crossed the empty valley and found}"
export PLANNING_GOAL_2="${PLANNING_GOAL_2:-a hidden village.}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}" DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-131072}" NUM_ITERATIONS="${NUM_ITERATIONS:-6000}"
export EVAL_EVERY="${EVAL_EVERY:--1}" CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}" SAMPLE_EVERY="${SAMPLE_EVERY:--1}"
export SAVE_EVERY="${SAVE_EVERY:-500}" CHECKPOINT_KEEP=1 POST_EVAL=0 RUN_SFT=0
export XLA_SYNC_MODE="${XLA_SYNC_MODE:-microstep-wait}" WANDB_RUN="${WANDB_RUN:-dummy}"
export SKIP_SETUP="${SKIP_SETUP:-1}" SKIP_DATA="${SKIP_DATA:-1}" MODEL_TAG="${MODEL_TAG:-llwm_tokenst_d15}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/start_tpu_v4_16_tmux.sh"
