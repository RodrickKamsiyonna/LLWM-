#!/usr/bin/env bash
# LLWM depth sweep. DEPTHS='4 8 12' changes the series; TPU is the default.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
setup_run
prepare_data
SERIES_NAME="${1:-${SERIES_NAME:-llwm_$(date +%b%d)}}"
read -r -a SERIES_DEPTHS <<< "${DEPTHS:-4 6 8 12}"
RESULTS_DIR="$NANOCHAT_BASE_DIR/${SERIES_NAME}_miniseries_results"
run_cmd mkdir -p "$RESULTS_DIR"
for d in "${SERIES_DEPTHS[@]}"; do
    TAG="${SERIES_NAME}_d${d}"
    START_TIME=$(date +%s)
    launch scripts.base_train "${LLWM_ARGS[@]}" "${PLAN_ARGS[@]}" \
        --depth="$d" --head-dim="$HEAD_DIM" --max-seq-len="$MAX_SEQ_LEN" \
        --device-batch-size="$DEVICE_BATCH_SIZE" --total-batch-size="$TOTAL_BATCH_SIZE" \
        --num-iterations="${NUM_ITERATIONS:--1}" --target-param-data-ratio="${PARAM_DATA_RATIO:-8}" \
        --run="${WANDB_RUN}_${TAG}" --model-tag="$TAG" \
        --eval-every="$EVAL_EVERY" --eval-tokens="$EVAL_TOKENS" \
        --core-metric-every="${SWEEP_CORE_EVERY:-999999}" --core-metric-max-per-task="$CORE_MAX_PER_TASK" \
        --sample-every="$SAMPLE_EVERY" --eval-max-new-tokens="$EVAL_MAX_NEW_TOKENS" --save-every=-1
    END_TIME=$(date +%s)
    run_cmd python -m scripts.run_summary --model-tag="$TAG" --label="$SERIES_NAME" \
        --wall-seconds="$((END_TIME - START_TIME))" --results-file="$RESULTS_DIR/results.csv"
done
