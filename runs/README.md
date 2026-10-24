# Start LLWM training

Run from the nanochat repository directory. The recipes now default to a Linux
TPU VM with PyTorch/XLA, not the upstream eight-H100 configuration. They install
matching TPU dependencies in `.venv`, prepare CPU tokenizer/data artifacts, and
launch LLWM pretraining and SFT using PJRT. They do not create a TPU VM.

```bash
# Inspect the exact commands; no installation, data download or training occurs.
DRY_RUN=1 bash runs/speedrun.sh

# Start pretraining, evaluation, SFT, and SFT evaluation.
bash runs/speedrun.sh

# With W&B (authenticate first with `wandb login`).
WANDB_RUN=my_llwm bash runs/speedrun.sh

# Reuse prepared dependencies/data, with a short pretraining-only run.
SKIP_SETUP=1 SKIP_DATA=1 RUN_SFT=0 NUM_ITERATIONS=100 bash runs/speedrun.sh
```

All recipes use the LLWM objective and reference depth ratio. EQM is absent.
Default settings in [common.sh](common.sh):

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `NANOCHAT_DEVICE` | `tpu` | `tpu`/`xla`, `cuda`, `cpu`, or `mps` |
| `DEPTH` | `12` | Compressor layers; default predictor layers are 9 |
| `MAX_SEQ_LEN` | `512` | Training context length |
| `HEAD_DIM` | `64` | Attention head width |
| `DEVICE_BATCH_SIZE` | `1` | Training sequences per replica |
| `TOTAL_BATCH_SIZE` | `32768` | Global training tokens per update |
| `NUM_ITERATIONS` | `1000` in speedrun | Pretraining updates; `-1` selects the data/parameter ratio |
| `ACTION_DIM`, `ACTION_LAYERS` | `128`, `2` | Gaussian action bottleneck and encoder depth |
| `PREDICTOR_LAYERS` | automatic | Optional predictor-depth override |
| `TARGET_KL`, `KL_LAMBDA` | `1`, `1` | Per-token total KL budget and penalty weight |
| `CE_WEIGHT`, `MEMORY_WEIGHT` | `1`, `1` | Training objective weights |
| `MEMORY_LOSS` | `smooth_l1` | `smooth_l1` or `mse` |
| `EVAL_MODE` | `plan` | Planned evaluation; `prior` selects the diagnostic baseline |
| `PLAN_GOAL` | `.` | Fixed planning suffix for CORE, samples, and ChatCORE |
| `EVAL_MAX_NEW_TOKENS` | `16` | Pretraining CORE/sample token budget **and planning horizon** |
| `CHAT_MAX_NEW_TOKENS` | `128` | SFT/standalone ChatCORE token budget **and planning horizon** |
| `PLAN_ITERATIONS` | `20` | Action optimization updates per plan (CLI default is 100) |
| `PLAN_LR` | `0.05` | Action optimizer learning rate |
| `PLAN_CONSISTENCY_WEIGHT` | `1` | Inverse-dynamics consistency coefficient |
| `PLAN_SEED` | `123` | Independent Gaussian action initialization seed |
| `CORE_MAX_PER_TASK` | `16` | Questions per CORE task; `-1` selects all questions |
| `CORE_METRIC_EVERY` | `1000` | Pretraining planned CORE interval; also runs at the end |
| `SAMPLE_EVERY` | `250` | Planned text-sample interval; also runs at the end |
| `EVAL_EVERY`, `EVAL_TOKENS` | `250`, `32768` | Prior-BPB interval and token count |
| `SFT_NUM_ITERATIONS` | `500` | SFT updates |
| `CHATCORE_EVERY` | `250` | SFT planned ChatCORE interval; also runs at the end |
| `CHATCORE_MAX_CAT`, `CHATCORE_MAX_SAMPLE` | `16`, `8` | SFT categorical/generative question limits |
| `DATA_SHARDS` | `170` | Downloaded pretraining shards |
| `RUN_SFT` | `1` | Set `0` for pretraining/evaluation only |

For example, to use a 64-token horizon for both pretraining and SFT evaluations:

```bash
EVAL_MAX_NEW_TOKENS=64 CHAT_MAX_NEW_TOKENS=64 PLAN_ITERATIONS=20 \
  WANDB_RUN=llwm_plan64 bash runs/speedrun.sh
```

The period is an optimization goal, not a forced appended character or a known
benchmark answer. Every action starts independently from a standard normal.
The predictor rolls through latent memories; only actions are optimized, while
model weights stay frozen. The horizon always equals that call's allowed output
budget. A successful goal suffix does not imply a correct benchmark answer.

CORE keeps its normal prompts, few-shot demonstrations, candidate ranking and
answer checks. It scores planned logits instead of zero-action logits. If a
candidate continuation is longer than the output budget, it scores one planned
block at a time and conditions the next block only on the preceding tokens.
Answers are used for scoring, never as action optimization goals. Planned CORE
scores are marked as planned and are not the upstream standard CORE metric.
BPB and RL's on-policy training log-probabilities still use the prior baseline.

Planning runs many backward passes per evaluation question. These recipes limit
question counts and use 20 planning updates to keep initial runs smaller. Increase
question limits for a fuller evaluation. Long SFT horizons can produce large XLA
compilations. No TPU throughput or memory-fit guarantee has been measured here.
Global batch size must be divisible by
`device_batch_size * max_seq_len * world_size`; increase `TOTAL_BATCH_SIZE` for
large pods. Use the same prepared data/tokenizer and run the same launcher on each
host for a PJRT pod. These scripts configure launches, not multi-host provisioning.

## Other recipes

```bash
# Depth sweep, with the same LLWM and planning settings.
DEPTHS='4 8 12' bash runs/miniseries.sh my_series

# FLOP-budget sweep; records actual checkpoint metadata in CSV.
DEPTHS='4 8' FLOPS_BUDGETS='1e18 2.15e18' bash runs/scaling_laws.sh

# Small CPU recipe; NANOCHAT_DEVICE=mps selects MPS.
bash runs/runcpu.sh

# CUDA keeps torchrun; it does not use PJRT or TPU packages.
NANOCHAT_DEVICE=cuda NPROC_PER_NODE=8 bash runs/speedrun.sh
```

`DRY_RUN=1`, `SKIP_SETUP=1`, and `SKIP_DATA=1` work across these recipes. Sweeps
write CSV results from saved checkpoint metadata, including the actual global
batch, action/predictor parameter counts, and evaluation mode/horizon. The old
GPU timing estimates and grep-based summaries no longer apply.

See [LLWM.md](../LLWM.md) and [TPU.md](../TPU.md) for architecture and installation.
