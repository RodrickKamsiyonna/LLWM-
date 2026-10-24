#!/usr/bin/env bash
# Shared LLWM setup. Source from a run script after `set -euo pipefail`.
RUNS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUNS_DIR/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
NANOCHAT_DEVICE="${NANOCHAT_DEVICE:-tpu}"
WANDB_RUN="${WANDB_RUN:-dummy}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEPTH="${DEPTH:-12}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
HEAD_DIM="${HEAD_DIM:-64}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32768}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-16}"
CHAT_MAX_NEW_TOKENS="${CHAT_MAX_NEW_TOKENS:-128}"
PLAN_GOAL="${PLAN_GOAL:-.}"
PLAN_ITERATIONS="${PLAN_ITERATIONS:-20}"
PLAN_LR="${PLAN_LR:-0.05}"
PLAN_CONSISTENCY_WEIGHT="${PLAN_CONSISTENCY_WEIGHT:-1}"
PLAN_SEED="${PLAN_SEED:-123}"
EVAL_MODE="${EVAL_MODE:-plan}"
EVAL_TOKENS="${EVAL_TOKENS:-32768}"
CORE_MAX_PER_TASK="${CORE_MAX_PER_TASK:-16}"
EVAL_EVERY="${EVAL_EVERY:-250}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-1000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-250}"
DATA_SHARDS="${DATA_SHARDS:-170}"

LLWM_ARGS=(--model-type=llwm --action-dim="${ACTION_DIM:-128}" --action-layers="${ACTION_LAYERS:-2}"
    --target-kl="${TARGET_KL:-1}" --kl-lambda="${KL_LAMBDA:-1}" --ce-weight="${CE_WEIGHT:-1}"
    --memory-weight="${MEMORY_WEIGHT:-1}" --memory-loss="${MEMORY_LOSS:-smooth_l1}")
if [[ -n "${PREDICTOR_LAYERS:-}" ]]; then
    LLWM_ARGS+=(--predictor-layers="$PREDICTOR_LAYERS")
fi
PLAN_ARGS=(--eval-mode="$EVAL_MODE" --plan-goal="$PLAN_GOAL" --plan-iterations="$PLAN_ITERATIONS"
    --plan-lr="$PLAN_LR" --plan-consistency-weight="$PLAN_CONSISTENCY_WEIGHT" --plan-seed="$PLAN_SEED")

run_cmd() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

setup_run() {
    case "$NANOCHAT_DEVICE" in
        tpu|xla) DEVICE_TYPE=xla; INSTALL_EXTRA=tpu ;;
        cuda|gpu) DEVICE_TYPE=cuda; INSTALL_EXTRA=gpu ;;
        cpu|mps) DEVICE_TYPE="$NANOCHAT_DEVICE"; INSTALL_EXTRA=cpu ;;
        *) echo "NANOCHAT_DEVICE must be tpu, cuda, cpu or mps" >&2; return 1 ;;
    esac
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo "# LLWM device=$NANOCHAT_DEVICE; goal=$PLAN_GOAL; horizon=$EVAL_MAX_NEW_TOKENS; planning updates=$PLAN_ITERATIONS"
        return
    fi
    mkdir -p "$NANOCHAT_BASE_DIR"
    if [[ "${SKIP_SETUP:-0}" != 1 ]]; then
        if ! command -v uv >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
        fi
        command -v uv >/dev/null 2>&1 || { echo "Install uv and put it on PATH" >&2; return 1; }
        [[ -d .venv ]] || uv venv --python "${PYTHON_VERSION:-3.12}"
        uv sync --extra "$INSTALL_EXTRA"
    fi
    [[ -f .venv/bin/activate ]] || { echo "Missing .venv; unset SKIP_SETUP to install dependencies" >&2; return 1; }
    source .venv/bin/activate
}

launch() {
    local module="$1"
    shift
    case "$DEVICE_TYPE" in
        xla) run_cmd python -m scripts.xla_launch "$module" --device-type=xla "$@" ;;
        cuda) run_cmd torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m "$module" --device-type=cuda "$@" ;;
        *) run_cmd python -m "$module" --device-type="$DEVICE_TYPE" "$@" ;;
    esac
}

prepare_data() {
    [[ "${SKIP_DATA:-0}" == 1 ]] && return 0
    # Data and tokenizer preparation stay on CPU, outside the PJRT launcher.
    run_cmd env -u PJRT_DEVICE python -m nanochat.dataset -n 8
    run_cmd env -u PJRT_DEVICE python -m scripts.tok_train --max-chars="${TOKENIZER_MAX_CHARS:-2000000000}" --vocab-size="${VOCAB_SIZE:-32768}"
    run_cmd env -u PJRT_DEVICE python -m scripts.tok_eval
    run_cmd env -u PJRT_DEVICE python -m nanochat.dataset -n "$DATA_SHARDS"
}
