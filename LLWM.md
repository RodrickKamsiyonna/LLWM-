# Latent language world model

New base-training runs now default to `--model-type=llwm`. The model adapts the
four main modules from `LLWM_micro/llwm/model.py` to nanochat's PyTorch/XLA pipeline:

- **Compressor:** nanochat's causal transformer, token/value embeddings, rotary
  attention, and residual mechanisms, followed by a learned memory LayerNorm.
  `--depth` controls its transformer layer count.
- **Action encoder:** the reference's progressively halved MLP over a detached
  `(current_memory, next_memory)` pair, with separate Gaussian mean and log-std
  heads. Configure `--action-dim` (128 by default) and `--action-layers` (2).
- **Predictor:** the reference's action-conditioned adaptive LayerNorm residual
  MLP blocks and a final LayerNorm. There is no whole-predictor outer residual.
- **LLM head:** the vocabulary projection with `15 * tanh(logits / 15)` softcap.

The implementation is [nanochat/llwm.py](nanochat/llwm.py). The compressor reuses
nanochat's existing kernels and KV-cache interface rather than replacing them
with the micro repo's transformer. The reference's diagnostic memory probe and
EQM matching loss are not included.

## Depth balance

The default follows the reference's four compressor / three predictor balance:

```python
predictor_layers = ceil(3 * encoder_layers / 4)
```

| Compressor layers (`--depth`) | Predictor layers |
| --- | --- |
| 4 | 3 |
| 8 | 6 |
| 12 | 9 |
| 20 | 15 |

Use `--predictor-layers N` to override the automatic value. The resolved depths,
action settings, and objective settings are saved in checkpoint metadata.

## Training objective

```text
memory = compressor(input_tokens)
next_memory = detached compressor memory at the next token
mu, log_std = action_encoder(detach(memory), detach(next_memory))
action = mu + exp(clamp(log_std, -5, 2)) * Normal(0, I)
predicted_memory = predictor(memory, action)
logits = llm_head(predicted_memory)
loss = ce_weight * CE(logits, targets)
     + memory_weight * regression(predicted_memory, next_memory)
     + kl_lambda * relu(mean_total_action_KL - target_kl)^2
```

CE and memory regression train all four main modules through the current-memory
path. The posterior inputs and next-memory targets are detached. Regression is
Smooth L1 by default; `--memory-loss=mse` selects MSE. Losses mask target `-1`
and return finite zeros for entirely ignored targets. The KL budget applies to
mean total latent KL per valid token in each microbatch, in nats. `--target-kl`
defaults to 1.0; tune it for your task. No EQM loss is computed.

To supervise the final transition, the extra target token is compressed with its
full prefix under `no_grad`. This avoids mutating training KV caches on XLA, at
the cost of an extra compressor forward pass. Parameter/FLOP accounting includes
the additional modules and approximates this extra compute. The inherited GPU
scaling-law/LR heuristics have not been retuned for LLWM.

## TPU commands

Install the TPU dependencies and prepare tokenizer/data as described in
[TPU.md](TPU.md), then run from this repository's root:

```bash
# Synthetic training + RL + replica agreement; no data/tokenizer files needed.
python -m scripts.xla_launch scripts.llwm_smoke

# Example training run; size batch and depth to fit your TPU.
python -m scripts.xla_launch scripts.base_train \
  --model-type=llwm --depth=4 --head-dim=64 --max-seq-len=128 \
  --action-dim=128 --action-layers=2 --target-kl=1 \
  --device-batch-size=1 --total-batch-size=8192 --num-iterations=100 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --run=dummy

# SFT inherits architecture/objectives from the selected checkpoint.
python -m scripts.xla_launch scripts.chat_sft --model-tag=llwm_d4 --run=dummy
```

The global token batch must be divisible by
`device_batch_size * max_seq_len * world_size`. New tags default to `llwm_dN`;
select tags explicitly when mixing GPT and LLWM checkpoints. Resume with the
same architecture, objective settings, and topology plus `--resume-from-step=N`.

## Gaussian action planning

[nanochat/planning.py](nanochat/planning.py) ports the micro repo's planner. It
compresses the prompt once, then **independently initializes every action**:

```python
actions = torch.randn(batch_size, horizon, action_dim)
```

The initialization is standard normal across batches, horizon steps and action
dimensions, seeded for reproducibility. On XLA it is sampled on CPU and transferred
to the TPU because XLA does not implement `torch.Generator(device='xla')`.

Planning rolls forward through **predicted memories**, not generated-token
feedback. It optimizes only the actions against the requested goal suffix's CE
and the reference's inverse-dynamics consistency penalty. Consistency differentiates
through the action encoder and all rolled-out transitions, while model parameters
are frozen. Original training mode and parameter flags are restored afterward.
This consistency penalty is separate from EQM matching, which is absent.

```bash
python -m scripts.xla_launch --single-process scripts.llwm_plan \
  --source=base --model-tag=llwm_d4 \
  --prompt='A journey begins' --goal=' arrives home.' \
  --horizon=32 --iterations=100 --seed=123
```

The horizon must cover the goal's **BPE token** length. Goal exact-match measures
this goal-conditioned plan, not ordinary language-model evaluation accuracy.
Planning currently uses one device. Large horizons unroll large autograd/XLA
graphs; begin with short plans.

## Training-time CORE, ChatCORE and samples

LLWM evaluation now defaults to `--eval-mode=auto`, which selects planning for
LLWM and ordinary prior scoring for GPT. `--eval-mode=plan` explicitly requires
LLWM; `--eval-mode=prior` selects the zero-action diagnostic baseline.

CORE and text samples use the fixed `--plan-goal='.'`. Their planning horizon is
`--eval-max-new-tokens` (16 by default in pretraining/base eval). SFT ChatCORE
uses its configured output-token budget (512 by default in the CLI, 128 in the
run recipes). Goal text, planning iterations, learning rate, consistency weight,
and seed are configurable and saved with the training run settings.

CORE retains normal benchmark prompts, few-shot examples and continuation
scoring. Candidate answers are never planning goals. Continuations longer than
one budget are scored in planned blocks, conditioning on completed prefix tokens
only. Shared candidate prefixes reuse plans. ChatCORE categorical answers use the
first planned token's answer-letter logits; generative tasks use the planned
rollout. These scores are labeled as planned, rather than upstream standard CORE.
Latent plans do not execute Python tool calls.

The shared sampling adapter optimizes actions rather than sampling token logits;
its `temperature` and `top_k` arguments do not change action initialization.
Multiple samples use independent Gaussian actions and argmax token readouts.
Standalone `scripts.llwm_plan` is a single-process command. Distributed CORE and
ChatCORE shard questions across replicas and plan independently on each replica.

See [runs/README.md](runs/README.md) for ready-to-run TPU, CUDA and CPU recipes,
all environment settings, and previews using `DRY_RUN=1`.

## BPB, ordinary chat and RL

BPB remains a target-free zero-action baseline; it is not the posterior training
objective. Interactive `scripts.chat_cli` and RL's existing generation policy
also retain that baseline. `loss_reduction='none'` returns prior-policy token NLL,
so RL's sampled policy and scored probabilities agree. Posterior optimizer groups
are skipped during RL. Pretraining and SFT use the full world-model objective.

Legacy GPT checkpoints still load through the architecture-aware factory; new
GPT runs use `--model-type=gpt`. Existing GPT or LLWM_micro weights and optimizer
states are not automatically converted into this architecture.

## Validation

```bash
python -m pytest tests/test_llwm.py tests/test_xla_backend.py -q
python -m scripts.xla_launch scripts.llwm_smoke
```

CPU tests cover the ratio, memory alignment, detach boundaries, loss masking,
module gradients, optimizer coverage, causal prior scoring, checkpoint rebuilding,
and Gaussian action planning. The development Intel Mac uses torch 2.2 with
mathematically equivalent test-only RMSNorm/GQA shims; production pins torch/XLA
2.9. No TPU hardware run has been performed in this development environment.
