#!/usr/bin/env bash
# LLWM compute/depth sweep, with planned CORE scored at each run's final step.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
setup_run
prepare_data
LABEL="${LABEL:-llwm_$(date +%b%d)}"
read -r -a BUDGETS <<< "${FLOPS_BUDGETS:-1e18 2.15e18 4.64e18 1e19}"
read -r -a SWEEP_DEPTHS <<< "${DEPTHS:-4 6 8 12}"
RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_laws_results_${LABEL}"
run_cmd mkdir -p "$RESULTS_DIR"
for flops in "${BUDGETS[@]}"; do
    for d in "${SWEEP_DEPTHS[@]}"; do
        TAG="${LABEL}_${flops}_d${d}"
        START_TIME=$(date +%s)
        launch scripts.base_train "${LLWM_ARGS[@]}" "${PLAN_ARGS[@]}" \
            --depth="$d" --head-dim="$HEAD_DIM" --max-seq-len="$MAX_SEQ_LEN" \
            --device-batch-size="$DEVICE_BATCH_SIZE" --total-batch-size="$TOTAL_BATCH_SIZE" \
            --target-flops="$flops" --target-param-data-ratio=-1 \
            --run="${WANDB_RUN}_${TAG}" --model-tag="$TAG" \
            --eval-every="$EVAL_EVERY" --eval-tokens="$EVAL_TOKENS" \
            --core-metric-every="${SWEEP_CORE_EVERY:-999999}" --core-metric-max-per-task="$CORE_MAX_PER_TASK" \
            --sample-every="$SAMPLE_EVERY" --eval-max-new-tokens="$EVAL_MAX_NEW_TOKENS" --save-every=-1
        END_TIME=$(date +%s)
        run_cmd python -m scripts.run_summary --model-tag="$TAG" --label="$LABEL" \
            --flops-budget="$flops" --wall-seconds="$((END_TIME - START_TIME))" --results-file="$RESULTS_DIR/results.csv"
    done
done
