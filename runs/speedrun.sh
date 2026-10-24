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
launch scripts.base_train "${LLWM_ARGS[@]}" "${PLAN_ARGS[@]}" \
    --depth="$DEPTH" --head-dim="$HEAD_DIM" --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" --total-batch-size="$TOTAL_BATCH_SIZE" \
    --num-iterations="${NUM_ITERATIONS:-1000}" --target-param-data-ratio="${PARAM_DATA_RATIO:-8}" \
    --eval-every="$EVAL_EVERY" --eval-tokens="$EVAL_TOKENS" \
    --core-metric-every="$CORE_METRIC_EVERY" --core-metric-max-per-task="$CORE_MAX_PER_TASK" \
    --sample-every="$SAMPLE_EVERY" --eval-max-new-tokens="$EVAL_MAX_NEW_TOKENS" \
    --save-every="${SAVE_EVERY:-500}" --model-tag="$MODEL_TAG" --run="$WANDB_RUN"
launch scripts.base_eval "${PLAN_ARGS[@]}" --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" --split-tokens="$EVAL_TOKENS" \
    --max-per-task="$CORE_MAX_PER_TASK" --eval-max-new-tokens="$EVAL_MAX_NEW_TOKENS" \
    --unconditioned-max-new-tokens="$EVAL_MAX_NEW_TOKENS"
if [[ "${RUN_SFT:-1}" == 1 ]]; then
    launch scripts.chat_sft "${PLAN_ARGS[@]}" --model-tag="$MODEL_TAG" \
        --num-iterations="${SFT_NUM_ITERATIONS:-500}" --run="$WANDB_RUN" \
        --eval-every="$EVAL_EVERY" --eval-tokens="$EVAL_TOKENS" \
        --chatcore-every="${CHATCORE_EVERY:-250}" --chatcore-max-cat="${CHATCORE_MAX_CAT:-16}" \
        --chatcore-max-sample="${CHATCORE_MAX_SAMPLE:-8}" --eval-max-new-tokens="$CHAT_MAX_NEW_TOKENS"
    launch scripts.chat_eval "${PLAN_ARGS[@]}" -i sft --model-tag="$MODEL_TAG" \
        --eval-max-new-tokens="$CHAT_MAX_NEW_TOKENS" --max-problems="${CHATCORE_MAX_SAMPLE:-8}"
fi
