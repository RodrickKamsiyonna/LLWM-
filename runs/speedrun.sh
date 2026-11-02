#!/usr/bin/env bash
# TPU LLWM pretraining + SFT. This is no longer the upstream 8xH100 timing recipe.
# Preview: DRY_RUN=1 bash runs/speedrun.sh
# Start:   WANDB_RUN=my_llwm bash runs/speedrun.sh
# CUDA:   NANOCHAT_DEVICE=cuda bash runs/speedrun.sh
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
setup_run
prepare_data
MODEL_TAG="${MODEL_TAG:-llwm_d${DEPTH}}"
RELIABILITY_ARGS=(--xla-sync-mode="${XLA_SYNC_MODE:-microstep}" --xla-metrics-every="${XLA_METRICS_EVERY:-25}"
    --tokenizer-threads="${TOKENIZER_THREADS:-4}" --tokenizer-batch-size="${TOKENIZER_BATCH_SIZE:-128}"
    --document-buffer-size="${DOCUMENT_BUFFER_SIZE:-1000}" --max-buffer-tokens="${MAX_BUFFER_TOKENS:-4000000}"
    --max-document-chars="${MAX_DOCUMENT_CHARS:-1000000}" --max-document-tokens="${MAX_DOCUMENT_TOKENS:-262144}"
    --max-tokenizer-chars="${MAX_TOKENIZER_CHARS:-1000000}" --long-document-policy="${LONG_DOCUMENT_POLICY:-error}"
    --packing-policy="${PACKING_POLICY:-legacy_crop}" --checkpoint-keep="${CHECKPOINT_KEEP:-1}"
    --planning-every="${PLANNING_EVERY:--1}" --planning-prompt="${PLANNING_PROMPT:-The capital of France is}"
    --stop-after-step="${STOP_AFTER_STEP:--1}")
[[ -z "${DATASET_MANIFEST:-}" ]] || RELIABILITY_ARGS+=(--dataset-manifest="$DATASET_MANIFEST")
[[ -z "${RUN_ID:-}" ]] || RELIABILITY_ARGS+=(--run-id="$RUN_ID")
[[ -z "${CHECKPOINT_DIR:-}" ]] || RELIABILITY_ARGS+=(--checkpoint-dir="$CHECKPOINT_DIR")
[[ -z "${LOG_DIR:-}" ]] || RELIABILITY_ARGS+=(--log-dir="$LOG_DIR")
[[ "${CHECKPOINT_SHARED:-0}" != 1 ]] || RELIABILITY_ARGS+=(--checkpoint-shared)
[[ "${DEBUG_DIAGNOSTICS:-0}" != 1 ]] || RELIABILITY_ARGS+=(--debug-diagnostics)
[[ "${VERIFY_COLLECTIVES:-0}" != 1 ]] || RELIABILITY_ARGS+=(--verify-collectives)
if [[ "${RESUME_EXTEND_SCHEDULE:-0}" == 1 && "${RESUME_LATEST:-0}" != 1 ]]; then
    echo "RESUME_EXTEND_SCHEDULE=1 requires RESUME_LATEST=1" >&2
    exit 2
fi
if [[ "${RESUME_LATEST:-0}" == 1 ]]; then
    RELIABILITY_ARGS+=(--resume-latest)
    [[ "${RESUME_EXTEND_SCHEDULE:-0}" != 1 ]] || RELIABILITY_ARGS+=(--resume-extend-schedule)
elif [[ -n "${RESUME_FROM_STEP:-}" ]]; then
    RELIABILITY_ARGS+=(--resume-from-step="$RESUME_FROM_STEP")
fi
launch scripts.base_train "${LLWM_ARGS[@]}" "${PLAN_ARGS[@]}" "${RELIABILITY_ARGS[@]}" \
    --depth="$DEPTH" --aspect-ratio="$ASPECT_RATIO" --head-dim="$HEAD_DIM" --window-pattern="${WINDOW_PATTERN:-SSSL}" --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" --total-batch-size="$TOTAL_BATCH_SIZE" \
    --num-iterations="${NUM_ITERATIONS:-1000}" --target-param-data-ratio="${PARAM_DATA_RATIO:-8}" \
    --eval-every="$EVAL_EVERY" --eval-tokens="$EVAL_TOKENS" \
    --core-metric-every="$CORE_METRIC_EVERY" --core-metric-max-per-task="$CORE_MAX_PER_TASK" \
    --sample-every="$SAMPLE_EVERY" --eval-max-new-tokens="$EVAL_MAX_NEW_TOKENS" \
    --save-every="${SAVE_EVERY:-500}" --model-tag="$MODEL_TAG" --run="$WANDB_RUN"
# Expensive/unstable evaluation is a separate opt-in stage, never forced by disabled intervals.
export NANOCHAT_CHECKPOINT_DIR="${CHECKPOINT_DIR:-$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/$RUN_ID}"
if [[ "${POST_EVAL:-0}" == 1 ]]; then
    launch scripts.base_eval "${PLAN_ARGS[@]}" --model-tag="$MODEL_TAG" \
        --device-batch-size="$DEVICE_BATCH_SIZE" --split-tokens="$EVAL_TOKENS" \
        --max-per-task="$CORE_MAX_PER_TASK" --eval-max-new-tokens="$EVAL_MAX_NEW_TOKENS" \
        --unconditioned-max-new-tokens="$EVAL_MAX_NEW_TOKENS"
fi
if [[ "${RUN_SFT:-1}" == 1 ]]; then
    launch scripts.chat_sft "${PLAN_ARGS[@]}" --model-tag="$MODEL_TAG/$RUN_ID" \
        --num-iterations="${SFT_NUM_ITERATIONS:-500}" --run="$WANDB_RUN" \
        --eval-every="$EVAL_EVERY" --eval-tokens="$EVAL_TOKENS" \
        --chatcore-every="${CHATCORE_EVERY:-250}" --chatcore-max-cat="${CHATCORE_MAX_CAT:-16}" \
        --chatcore-max-sample="${CHATCORE_MAX_SAMPLE:-8}" --eval-max-new-tokens="$CHAT_MAX_NEW_TOKENS"
    unset NANOCHAT_CHECKPOINT_DIR
    launch scripts.chat_eval "${PLAN_ARGS[@]}" -i sft --model-tag="$MODEL_TAG/$RUN_ID" \
        --eval-max-new-tokens="$CHAT_MAX_NEW_TOKENS" --max-problems="${CHATCORE_MAX_SAMPLE:-8}"
fi
