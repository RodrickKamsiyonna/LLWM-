#!/usr/bin/env bash
# Shared LLWM setup. Source from a run script after `set -euo pipefail`.
RUNS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUNS_DIR/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
NANOCHAT_DEVICE="${NANOCHAT_DEVICE:-tpu}"
WANDB_RUN="${WANDB_RUN:-dummy}"
export RUN_ID="${RUN_ID:-llwm-$(date -u +%Y%m%dT%H%M%SZ)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEPTH="${DEPTH:-12}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
HEAD_DIM="${HEAD_DIM:-64}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32768}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-16}"
CHAT_MAX_NEW_TOKENS="${CHAT_MAX_NEW_TOKENS:-128}"
PLAN_GOAL="${PLAN_GOAL:-.}"
PLAN_ITERATIONS="${PLAN_ITERATIONS:-20}"
PLAN_LR="${PLAN_LR:-0.05}"
PLAN_EXECUTE_STEPS="${PLAN_EXECUTE_STEPS:-5}"
PLAN_SEED="${PLAN_SEED:-123}"
EVAL_MODE="${EVAL_MODE:-plan}"
EVAL_TOKENS="${EVAL_TOKENS:-32768}"
CORE_MAX_PER_TASK="${CORE_MAX_PER_TASK:-16}"
EVAL_EVERY="${EVAL_EVERY:-250}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-1000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-250}"
DATA_SHARDS="${DATA_SHARDS:-170}"

LLWM_ARGS=(--embedding-variance-weight="${EMBEDDING_VARIANCE_WEIGHT:-0}" --embedding-std-floor="${EMBEDDING_STD_FLOOR:-0.8}" --token-objective="${TOKEN_OBJECTIVE:-ce}" --head-layers="${HEAD_LAYERS:-2}" --sigreg-weight="${SIGREG_WEIGHT:-0}"
    --sigreg-projections="${SIGREG_PROJECTIONS:-1024}" --sigreg-knots="${SIGREG_KNOTS:-17}" --model-type=llwm --action-dim="${ACTION_DIM:-8}" --action-layers="${ACTION_LAYERS:-2}"
    --prediction-horizon="${PREDICTION_HORIZON:-4}" --head-training="${HEAD_TRAINING:-detached}" --head-source="${HEAD_SOURCE:-compressor}"
    --action-conditioning="${ACTION_CONDITIONING:-memory_pair}" --planning-mode="${PLANNING_MODE:-predictor}"
    --target-kl="${TARGET_KL:-1}" --kl-lambda="${KL_LAMBDA:-1}" --kl-mode="${KL_MODE:-target}" --ce-weight="${CE_WEIGHT:-1}" --ce-threshold="${CE_THRESHOLD:-0}" --ce-power="${CE_POWER:-1}"
    --memory-weight="${MEMORY_WEIGHT:-1}" --memory-loss="${MEMORY_LOSS:-smooth_l1}")
if [[ -n "${PREDICTOR_LAYERS:-}" ]]; then
    LLWM_ARGS+=(--predictor-layers="$PREDICTOR_LAYERS")
fi
if [[ "${TRAIN_EMBEDDING_TARGET:-0}" == 1 ]]; then
    LLWM_ARGS+=(--train-embedding-target)
fi
PLAN_ARGS=(--eval-mode="$EVAL_MODE" --plan-goal="$PLAN_GOAL" --plan-iterations="$PLAN_ITERATIONS"
    --plan-lr="$PLAN_LR" --plan-execute-steps="$PLAN_EXECUTE_STEPS" --plan-seed="$PLAN_SEED")

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
        cuda|gpu) DEVICE_TYPE=cuda; INSTALL_EXTRA=gpu; CHECKPOINT_SHARED="${CHECKPOINT_SHARED:-1}" ;;
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
        INSTALL_ARGS=(--extra "$INSTALL_EXTRA")
        if [[ "${CHECKPOINT_DIR:-}${CONTROL_DIR:-}" == *gs://* ]]; then
            INSTALL_ARGS+=(--extra cloud)
        fi
        uv sync --python "${PYTHON_VERSION:-3.12}" "${INSTALL_ARGS[@]}"
    fi
    [[ -f .venv/bin/activate ]] || { echo "Missing .venv; unset SKIP_SETUP to install dependencies" >&2; return 1; }
    source .venv/bin/activate
}

launch() {
    local module="$1"
    shift
    # The sweep recipes use the same startup/storage guards as speedrun.
    if [[ "$module" == scripts.base_train ]]; then
        local -a startup_args=(--run-id="$RUN_ID")
        [[ -z "${DATASET_MANIFEST:-}" ]] || startup_args+=(--dataset-manifest="$DATASET_MANIFEST")
        [[ "${CHECKPOINT_SHARED:-0}" != 1 ]] || startup_args+=(--checkpoint-shared)
        if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
            startup_args+=(--checkpoint-dir="$CHECKPOINT_DIR")
        elif [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
            local tag="${MODEL_TAG:-llwm_d$DEPTH}" arg
            for arg in "$@"; do
                [[ "$arg" != --model-tag=* ]] || tag="${arg#--model-tag=}"
            done
            startup_args+=(--checkpoint-dir="${CHECKPOINT_ROOT%/}/$tag/$RUN_ID")
        fi
        set -- "${startup_args[@]}" "$@"
    fi
    case "$DEVICE_TYPE" in
        xla)
            LAUNCH_ARGS=()
            [[ -z "${RUN_ID:-}" ]] || LAUNCH_ARGS+=(--run-id="$RUN_ID")
            [[ -z "${ATTEMPT_ID:-}" ]] || LAUNCH_ARGS+=(--launch-id="$ATTEMPT_ID")
            [[ -z "${CONTROL_DIR:-}" ]] || LAUNCH_ARGS+=(--control-dir="$CONTROL_DIR")
            [[ -z "${LOG_DIR:-}" ]] || LAUNCH_ARGS+=(--log-dir="$LOG_DIR")
            [[ -z "${EXPECTED_WORLD_SIZE:-}" ]] || LAUNCH_ARGS+=(--expected-world-size="$EXPECTED_WORLD_SIZE")
            [[ -z "${EXPECTED_LOCAL_DEVICES:-}" ]] || LAUNCH_ARGS+=(--expected-local-devices="$EXPECTED_LOCAL_DEVICES")
            [[ "${DEBUG_DIAGNOSTICS:-0}" != 1 ]] || LAUNCH_ARGS+=(--debug)
            run_cmd python -m scripts.xla_launch "${LAUNCH_ARGS[@]}" "$module" --device-type=xla "$@"
            ;;
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

if [[ "${EMBEDDING_DISTANCE_CE:-0}" == "1" ]]; then LLWM_ARGS+=(--embedding-distance-ce); fi
