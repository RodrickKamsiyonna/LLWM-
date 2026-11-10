# Current Kaggle embedding-MSE configuration

Use `notebooks/kaggle_embedding_mse/llwm_embedding_mse.ipynb` or `runs/kaggle_embedding_mse.sh` on a single-host TPU v5e-8. The model compresses causal context, infers Gaussian actions from current memory plus the observed next-token embedding, and predicts the next-token embedding through six adaptive predictor blocks and an MLP/embedding head. Training minimizes masked embedding MSE against a detached target embedding plus `3 * relu(KL_total - 1)^2`. Embeddings remain trainable through context and posterior input paths. The optimizer explicitly covers the learned embedding head.

MPC rolls continuous embedding predictions through the compressor and differentiable KV/smear cache. It minimizes MSE against goal-token embeddings; model weights are frozen and only actions are optimized. At execution boundaries, nearest-embedding tokens are committed and context is recompressed. The best observed action sequence is retained. Default horizon20/execute10,100 iterations,LR0.001; two tasks every1000 training updates. There is no memory regression or SIGReg in this profile.

This configuration is prepared for Kaggle but requires an actual eight-device run to validate memory, collectives and throughput. The notebook runs a tiny distributed embedding-MSE smoke test before full training. Full model settings are compressor15,width1664,predictor6,action128,context1024,global batch131072,6000 updates. Do not silently reduce architecture or token budget if it runs out of memory.

The older configurations and their semantics are documented below for checkpoint compatibility.

# Latent language world model

## Token-conditioned posterior and straight-through planning

The new profile is `runs/token_st_tpu_v4_16.sh`. It uses 15 compressor layers,
6 predictor layers and 128-dimensional actions. The posterior is
`q(a | compressor(context), embedding(next_token))`, with joint gradients.
The predictor conditions on `(h, a)` and its MLP/vocabulary head predicts the
next token. Training teacher-forces all observed next tokens and uses
`CE + 3 * relu(total_KL - 1)**2`. There is no memory regression, no detached
future compressor target and no extra future encoder pass. Accordingly,
`PREDICTION_HORIZON=1` describes training; it does not limit planning's horizon.

With `PLANNING_MODE=straight_through`, each imagined token is selected by argmax,
then re-encoded by the compressor. Both token and value embeddings use the exact
hard selection in the forward pass and an expected-embedding surrogate in the
backward pass. The planner minimizes goal-suffix CE through this full loop.
A functional KV cache reuses prompt and generated history without in-place
tensor updates. Each optimization iteration forks the immutable prompt cache;
step rematerialization bounds saved activations. Model parameters stay frozen.
MPC commits tokens using ordinary hard embedding lookups, updates the real cache,
then warm-starts the remaining actions. The default profile plans every 500
training steps, with horizon 20, execute 10, 100 iterations and LR 0.01.

`scripts.llwm_token_smoke` exercises training and straight-through MPC on all
eight devices across both TPU hosts. `tests/test_token_conditioned.py` checks
posterior alignment, hard-forward equivalence, cached recompression, gradients
through earlier actions, rematerialization, masking and checkpoint rebuilding.

The sections below also describe earlier architectures, retained for loading
existing checkpoints through their saved configuration.

New base-training runs now default to `--model-type=llwm`. The model adapts the
four main modules from `LLWM_micro/llwm/model.py` to nanochat's PyTorch/XLA pipeline:

- **Compressor:** nanochat's causal transformer, token/value embeddings, rotary
  attention, and residual mechanisms, followed by a learned memory LayerNorm.
  `--depth` controls its transformer layer count.
- **Action encoder:** the reference's progressively halved MLP over a detached
  `(current_memory, next_memory)` pair, with separate Gaussian mean and log-std
  heads. Configure `--action-dim` (8 by default) and `--action-layers` (2).
- **Predictor:** the reference's action-conditioned adaptive LayerNorm residual
  MLP blocks and a final LayerNorm. There is no whole-predictor outer residual.
- **LLM head:** with `--head-source=action`, project the action to model width,
  apply a two-layer MLP, then vocabulary projection and `15 * tanh(logits / 15)`.

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
predicted[1] = predictor(memory, action)
predicted[j] = predictor(predicted[j-1], next_action)
token_logits = vocabulary_head(MLP(action_projection(action)))
ce = mean CE(token_logits, next_token)
for j in 1..prediction_horizon:
    memory regression(predicted[j], next_memory[j])
loss = ce_weight * ce + memory_weight * memory_regression
       + kl_lambda * relu(total_action_KL - target_kl)**2
```

`--prediction-horizon` defaults to 4. For each starting memory, the predictor
recursively consumes the next inferred actions and predicts several future
memories. Memory errors are averaged over all valid start/offset pairs; CE and
KL are charged once per observed action. For action heads, `--head-training=joint`
lets CE train the action encoder, projection, MLP and vocabulary head. The
posterior's memory inputs remain detached, so this CE path does not directly
train the compressor or predictor; memory regression trains those modules.
`detached` stops CE gradients at the action input to the projection. Regression is Smooth L1 by
default; `--memory-loss=mse` selects MSE. Losses mask target `-1`. The KL budget
applies to mean total latent KL per valid token, in nats. `--kl-mode=standard`
instead directly minimizes weighted total KL. No EQM loss is computed.

`--head-source=predictor` retains next-token decoding from the first predicted
memory. `compressor` retains current-token reconstruction for older checkpoints.
Head source is checkpointed; old checkpoints do not acquire new projection weights.

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
  --action-dim=8 --action-layers=2 --prediction-horizon=4 --target-kl=1 \
  --device-batch-size=1 --total-batch-size=8192 --num-iterations=100 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --run=dummy

# SFT inherits architecture/objectives from the selected checkpoint.
python -m scripts.xla_launch scripts.chat_sft --model-tag=llwm_d4 --run=dummy
```

The global token batch must be divisible by
`device_batch_size * max_seq_len * world_size`. New tags default to `llwm_dN`;
select tags explicitly when mixing GPT and LLWM checkpoints. Resume with the
same architecture, objective settings, and topology plus `--resume-from-step=N`.

## Predictor-only MPC planning

[nanochat/planning.py](nanochat/planning.py) ports the micro repo's model
predictive control (MPC) planner. It compresses the real prompt once, then
**independently initializes every candidate action**:

```python
actions = torch.randn(batch_size, horizon, action_dim)
```

The initialization is standard normal across batches, horizon steps and action
dimensions, seeded for reproducibility. On XLA it is sampled on CPU and transferred
to the TPU because XLA does not implement `torch.Generator(device='xla')`.

With action heads, the compressor separately encodes the goal text once. The
planning objective is SmoothL1 between the terminal predicted memory and the
goal's final compressor memory. The goal never enters the real prompt KV cache.
Each imagined rollout uses only the predictor. After optimizing, the selected
actions are decoded through projection, MLP and vocabulary head.
MPC commits `--plan-execute-steps` tokens,
feeds those chosen tokens through the cached compressor to update the real
memory, then replans the remaining horizon. Older memory-head checkpoints retain
goal-suffix CE planning. The action encoder and consistency penalties are not used
for planning, and model parameters stay frozen.

```bash
python -m scripts.xla_launch --single-process scripts.llwm_plan \
  --source=base --model-tag=llwm_d4 \
  --prompt='A journey begins' --goal=' arrives home.' \
  --horizon=32 --execute-steps=5 --iterations=100 --seed=123
```

The horizon must cover the goal's **BPE token** length, and execute steps must
not exceed the horizon. MPC reports each round's remaining horizon and committed
token count. Goal exact-match measures this goal-conditioned plan, not ordinary
language-model evaluation accuracy.

## Training-time CORE, ChatCORE and samples

LLWM evaluation now defaults to `--eval-mode=auto`, which selects planning for
LLWM and ordinary prior scoring for GPT. `--eval-mode=plan` explicitly requires
LLWM; `--eval-mode=prior` selects the zero-action diagnostic baseline.

CORE and text samples use the fixed `--plan-goal='.'`. Their MPC horizon is
`--eval-max-new-tokens` (16 by default in pretraining/base eval), and
`--plan-execute-steps` (5 by default) controls how many tokens are committed
before replanning. SFT ChatCORE uses its configured output-token budget (512 by
default in the CLI, 128 in the run recipes). Goal text, planning iterations,
learning rate, execute-step count and seed are configurable and saved with the
training run settings.

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
For action heads the zero-action readout is independent of context and is only
a diagnostic; use MPC for contextual generation.

Legacy GPT checkpoints still load through the architecture-aware factory; new
GPT runs use `--model-type=gpt`. Existing GPT or LLWM_micro weights and optimizer
states are not automatically converted into this architecture.

## Validation

Straight-through token planning on XLA uses a stagewise backward pass. Each
token's local vector-Jacobian product propagates the latent, every KV cache
tensor, and the smear embedding adjoints to the preceding token. Execution
boundaries keep these backward graphs bounded without truncating gradients.
Model parameters remain frozen during action optimization. CPU regression tests
compare action gradients against the full unrolled computation, including a
nonzero smear coefficient and multi-token goals.

The first planning event compiles graphs for the prompt and cache lengths and
can take several minutes. Later iterations reuse those graphs. Rank diagnostics
record forward/backward/update durations separately, plus XLA compile and
execution counters; `Planning timing` lines expose completed iterations in the
stdout monitor. `scripts.llwm_plan_bench` benchmarks a saved checkpoint with its
optimizer buffers resident, without updating training weights or checkpoints.

```bash
python -m pytest tests/test_llwm.py tests/test_xla_backend.py -q
python -m scripts.xla_launch scripts.llwm_smoke
python -m scripts.xla_launch scripts.llwm_action_smoke
```

CPU tests cover the ratio, memory alignment, detach boundaries, loss masking,
module gradients, optimizer coverage, causal prior scoring, checkpoint rebuilding,
and Gaussian action planning. The development Intel Mac uses torch 2.2 with
mathematically equivalent test-only RMSNorm/GQA shims; production pins torch/XLA
2.9. The action-head smoke check also exercises two optimizer steps, replica
agreement, and goal-memory MPC (horizon 20, execute 10) on the real eight-device,
two-host TPU v4-16. This verifies execution and gradients, not trained planning quality.
