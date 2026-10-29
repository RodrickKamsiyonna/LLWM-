#!/usr/bin/env bash
# Run this same profile on BOTH hosts. No model/loss/token-budget simplification.
# RUN_ID must be identical on both hosts; ATTEMPT_ID changes when resuming.
set -euo pipefail
: "${RUN_ID:?Set one identical RUN_ID on both TPU hosts}"
: "${CHECKPOINT_ROOT:?Set an explicitly shared path or gs:// checkpoint root}"
: "${DATASET_MANIFEST:?Set this host successful full preflight manifest}"
export ATTEMPT_ID="${ATTEMPT_ID:-$RUN_ID}"
export NANOCHAT_DEVICE=tpu
export NANOCHAT_DTYPE=bfloat16
export EXPECTED_WORLD_SIZE=8
export EXPECTED_LOCAL_DEVICES=4
export DEPTH="${DEPTH:-15}"
export PREDICTOR_LAYERS="${PREDICTOR_LAYERS:-11}"
export ASPECT_RATIO="${ASPECT_RATIO:-110}"
export HEAD_DIM="${HEAD_DIM:-128}"
export WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
export ACTION_DIM="${ACTION_DIM:-128}"
export ACTION_LAYERS="${ACTION_LAYERS:-4}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-131072}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-6000}"
export MAX_BUFFER_TOKENS="${MAX_BUFFER_TOKENS:-64000000}"
# Explicit high limits preserve existing long documents; preflight must use these.
export MAX_DOCUMENT_CHARS="${MAX_DOCUMENT_CHARS:-16777216}"
export MAX_DOCUMENT_TOKENS="${MAX_DOCUMENT_TOKENS:-4194304}"
export MAX_TOKENIZER_CHARS="${MAX_TOKENIZER_CHARS:-16777216}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CHECKPOINT_ROOT%/}/$RUN_ID}"
export CONTROL_DIR="${CONTROL_DIR:-${CHECKPOINT_ROOT%/}/control}"
export CHECKPOINT_SHARED="${CHECKPOINT_SHARED:-1}"
export SAVE_EVERY="${SAVE_EVERY:-500}"
export CHECKPOINT_KEEP="${CHECKPOINT_KEEP:-1}"
export EVAL_EVERY="${EVAL_EVERY:--1}"
export CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
export SAMPLE_EVERY="${SAMPLE_EVERY:--1}"
export PLANNING_EVERY="${PLANNING_EVERY:--1}"
export POST_EVAL="${POST_EVAL:-0}"
export RUN_SFT="${RUN_SFT:-0}"
export SKIP_DATA="${SKIP_DATA:-1}"
export MODEL_TAG="${MODEL_TAG:-llwm_d15}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/speedrun.sh"
