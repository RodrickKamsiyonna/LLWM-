# TPU training and inference (PyTorch/XLA)

This backend uses PJRT replicated data parallelism, one model and optimizer replica
per XLA device. It covers base pretraining, SFT, RL, loss/CORE/chat evaluation,
checkpoint resume, CLI chat, and inference benchmarking. Tokenizer training and
Parquet preparation remain CPU jobs. CUDA/CPU/MPS entrypoints remain available.

## Install on a Linux TPU VM

Use Python 3.11 or 3.12 and a clean environment. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[tpu]'
```

The TPU extra pins torch and torch_xla to 2.9.0 together; the GPU/CPU extras retain
2.9.1. Do not combine extras. With uv, use `uv sync --extra tpu`; the lockfile includes the TPU dependency set.

## Prepare data once per host

```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval
```

Copy the same tokenizer and data to every host for a pod run. Each rank must have
at least one Parquet row group in both train and validation splits.

## Smoke run, then scale

```bash
python -m scripts.xla_launch --single-process scripts.base_train \
  --device-type=xla --depth=2 --head-dim=64 --max-seq-len=128 \
  --device-batch-size=1 --total-batch-size=128 --num-iterations=2 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --run=dummy
```

For replicated training across all local TPU devices:

```bash
python -m scripts.xla_launch scripts.base_train \
  --device-type=xla --depth=12 --device-batch-size=2 --run=dummy
```

Global token batch size must be divisible by
`device_batch_size * max_seq_len * world_size`. Start small: model parameters,
optimizer states, activations and explicit attention scores must fit on each
device. There is no optimizer sharding, tensor parallelism or SPMD in this backend.
FP8 remains CUDA-only. BF16 is selected automatically by the launcher.

Resume with the same topology and arguments plus `--resume-from-step=STEP`.
TPU optimizer checkpoints contain complete replicas, unlike CUDA optimizer shards;
do not warm-start TPU optimization from a distributed CUDA optimizer checkpoint.
Model weights are portable across backends. For SFT from CUDA weights, pass
`--load-optimizer=0`.

## Subsequent stages

```bash
python -m scripts.xla_launch scripts.base_eval --device-type=xla
python -m scripts.xla_launch scripts.chat_sft --device-type=xla --load-optimizer=0 --run=dummy
python -m scripts.xla_launch scripts.chat_eval --device-type=xla
python -m scripts.xla_launch scripts.chat_rl --device-type=xla --run=dummy
python -m scripts.xla_launch --single-process scripts.chat_cli --device-type=xla
python -m scripts.xla_launch --single-process scripts.infer_bench
```

Use each stage's `--help` for dataset, source checkpoint and step options. A tiny
smoke checkpoint is for validation, not useful chat quality. On a pod, launch the
same command simultaneously on **every host**, using PJRT's TPU VM configuration;
do not wrap this launcher in torchrun. Each host needs the same model checkpoint.
Use a shared checkpoint directory or copy rank-0 model/metadata files and all
optimizer rank files before resuming. Interactive chat and benchmarking use a
single host and single process.

## Validation and performance limits

```bash
pip install pytest
python -m pytest tests/test_xla_backend.py -k 'not tpu'
python -m scripts.xla_launch scripts.xla_smoke
PJRT_DEVICE=TPU python -m pytest tests/test_xla_backend.py -k tpu
```

The TPU test runs two optimizer steps, a checkpoint roundtrip and seeded generation
on one device. Multi-device/pod correctness and throughput still require hardware
validation. Five CPU numerical tests pass (attention parity and AdamW parity), along with
Python syntax, whitespace and dependency-lock checks. The CPU tests ran on
PyTorch 2.2.2, the newest wheel supported by the development Intel Mac; the
target runtime is PyTorch/XLA 2.9.0 on Linux. TPU tests have not been run.

The first steps compile and can be slow. Attention currently uses explicit matmuls
and masks, not a fused TPU attention kernel, so score memory grows quadratically
with sequence length. Generation uses the existing variable-length KV cache and
CPU sampling; changing context lengths can recompile graphs. RL rollouts and
variable-shaped evaluations have the same limitation. Benchmark results include
these costs; TPU peak-memory and roofline metrics are currently unavailable and
printed as zero. This is a functional port pending hardware validation, not a
TPU throughput claim.

References: [XLA execution and launch](https://docs.pytorch.org/xla/release/r2.8/learn/xla-overview.html),
[PyTorch/XLA releases](https://github.com/pytorch/xla/releases).

## LLWM architecture

New training runs now default to LLWM. See [LLWM.md](LLWM.md) for the
3:4 predictor/compressor depth ratio, objective settings and Gaussian action
planning. Use `--model-type=gpt` for legacy GPT training.

Ready-to-run TPU recipes, planning goals/horizons and configuration overrides
are listed in [runs/README.md](runs/README.md). Preview without launching training
using `DRY_RUN=1 bash runs/speedrun.sh`.
