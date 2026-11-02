"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
import socket
import uuid
from pathlib import Path
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPTConfig, Linear
from nanochat.llwm import LLWMConfig
from nanochat.model import create_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compile_model, synchronize_device, compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.reliable_checkpoint import TrainingCheckpoints, capture_rng, restore_rng
from nanochat.data_validation import DataOptions, validate_manifest, validate_ownership, sha256_file
from nanochat.distributed_utils import require_equal, accumulation_steps
from nanochat import diagnostics as diag
from nanochat.common import xla_microstep_boundary
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.planning_eval import add_planning_args, planning_for_model, PlanningEngine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps|xla (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--model-type", choices=["llwm", "gpt"], default="llwm", help="architecture for new training runs")
parser.add_argument("--predictor-layers", type=int, default=None, help="LLWM predictor depth (default: ceil(3 * depth / 4))")
parser.add_argument("--action-dim", type=int, default=128)
parser.add_argument("--action-layers", type=int, default=2)
parser.add_argument("--target-kl", type=float, default=1.0, help="LLWM total action KL budget per valid token, in nats")
parser.add_argument("--kl-lambda", type=float, default=1.0)
parser.add_argument("--ce-weight", type=float, default=1.0)
parser.add_argument("--memory-weight", type=float, default=1.0)
parser.add_argument("--memory-loss", choices=["smooth_l1", "mse"], default="smooth_l1")
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
add_planning_args(parser)
parser.add_argument('--run-id', default=os.environ.get('NANOCHAT_RUN_ID', os.environ.get('RUN_ID')))
parser.add_argument('--log-dir', default=os.environ.get('LOG_DIR', 'logs'))
parser.add_argument('--debug-diagnostics', action='store_true')
parser.add_argument('--watchdog-seconds', type=float, default=600)
parser.add_argument('--expected-world-size', type=int, default=0)
parser.add_argument('--verify-collectives', action='store_true')
parser.add_argument('--xla-sync-mode', choices=['step','microstep','microstep-wait'], default='microstep', help='async microstep boundary by default; blocking microstep-wait is diagnostic only')
parser.add_argument('--xla-metrics-every', type=int, default=25)
parser.add_argument('--dataset-manifest', help='this host full dataset preflight report')
parser.add_argument('--tokenizer-threads', type=int, default=4)
parser.add_argument('--tokenizer-batch-size', type=int, default=128)
parser.add_argument('--document-buffer-size', type=int, default=1000)
parser.add_argument('--max-buffer-tokens', type=int, default=4_000_000)
parser.add_argument('--max-document-chars', type=int, default=1_000_000)
parser.add_argument('--max-document-tokens', type=int, default=262144)
parser.add_argument('--max-tokenizer-chars', type=int, default=1_000_000)
parser.add_argument('--max-row-group-bytes', type=int, default=0)
parser.add_argument('--long-document-policy', choices=['error','chunk'], default='error')
parser.add_argument('--packing-policy', choices=['legacy_crop','retain_tail'], default='legacy_crop')
parser.add_argument('--checkpoint-dir', help='explicit full checkpoint run destination: shared filesystem or gs://')
parser.add_argument('--checkpoint-shared', action='store_true', help='assert filesystem destination is shared; visibility is verified on every rank')
parser.add_argument('--checkpoint-cache', help='per-host download cache (one optimizer download per host)')
parser.add_argument('--checkpoint-keep', type=int, default=1)
parser.add_argument('--resume-latest', action='store_true')
parser.add_argument('--resume-extend-schedule', action='store_true',
                    help='allow a longer second training phase from a checkpoint saved exactly at its prior schedule end')
parser.add_argument('--stop-after-step', type=int, default=-1, help='qualification cutoff; does NOT change the total schedule or token budget')
parser.add_argument('--planning-every', type=int, default=-1, help='independent periodic goal planning; -1 disables')
parser.add_argument('--planning-prompt', default='The capital of France is')
args = parser.parse_args()
args.run_id = args.run_id or ('local-' + uuid.uuid4().hex[:12])
if args.resume_latest and args.resume_from_step >= 0:
    parser.error('Use resume-latest OR resume-from-step')
if args.resume_extend_schedule and not args.resume_latest:
    parser.error('--resume-extend-schedule requires --resume-latest')
for name in ('eval_every','core_metric_every','sample_every','planning_every','save_every'):
    value=getattr(args,name)
    if value != -1 and value <= 0: parser.error(f'{name} must be -1 or positive')
print0('Effective CLI configuration:\n' + json.dumps(vars(args),sort_keys=True,indent=2))
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
expected_world=args.expected_world_size or int(os.environ.get('NANOCHAT_EXPECTED_WORLD_SIZE','0'))
if expected_world and ddp_world_size!=expected_world:
    raise ValueError(f'Expected world size {expected_world}, found {ddp_world_size}')
logs=os.environ.get('NANOCHAT_LAUNCH_LOGS') or str(Path(args.log_dir)/args.run_id/socket.gethostname())
diag.configure(logs,ddp_rank,ddp_local_rank,args.run_id,
               args.debug_diagnostics or os.environ.get('NANOCHAT_DEBUG')=='1',args.watchdog_seconds)
if args.verify_collectives: os.environ['NANOCHAT_VERIFY_COLLECTIVES']='1'
comparison_config={k:v for k,v in vars(args).items() if k not in ('dataset_manifest','log_dir','checkpoint_cache')}
require_equal(dict(config=comparison_config,world_size=ddp_world_size),'effective-training-configuration')
# Validate batch geometry before allocating model storage.
if args.total_batch_size > 0:
    accumulation_steps(args.total_batch_size,args.device_batch_size,args.max_seq_len,ddp_world_size)
data_options=DataOptions(tokenizer_threads=args.tokenizer_threads,tokenizer_batch_size=args.tokenizer_batch_size,
    buffer_size=args.document_buffer_size,max_buffer_tokens=args.max_buffer_tokens,
    max_document_chars=args.max_document_chars,max_document_tokens=args.max_document_tokens,
    max_tokenizer_chars=args.max_tokenizer_chars,long_document_policy=args.long_document_policy,
    packing_policy=args.packing_policy,max_row_group_bytes=args.max_row_group_bytes)
dataset_digest=None
if device.type=='xla' and ddp_world_size>1 and not args.dataset_manifest:
    raise ValueError('Multi-device TPU training requires --dataset-manifest from a full preflight on each host')
if args.dataset_manifest:
    from nanochat.dataset import list_parquet_files
    manifest=json.loads(Path(args.dataset_manifest).read_text())
    with diag.span('dataset_manifest_validation'):
        validate_manifest(manifest,list_parquet_files())
        validate_ownership(manifest,ddp_world_size)
        expected_policy=manifest['validation']['options']
        for name in ('max_document_chars','max_document_tokens','max_tokenizer_chars','long_document_policy','tokenizer_threads','tokenizer_batch_size'):
            if expected_policy[name]!=getattr(data_options,name):
                raise ValueError(f'Preflight and training data settings differ: {name}')
        for item in manifest['tokenizer']:
            path=Path(get_base_dir())/'tokenizer'/item['name']
            if sha256_file(path)!=item['sha256']:
                raise ValueError(f'Tokenizer changed since preflight: {path}')
        dataset_digest=manifest['digest']
        require_equal(dataset_digest,'dataset-and-tokenizer-inventory')
diag.event('effective_config',configuration=vars(args),world_size=ddp_world_size,
           data_options=data_options.to_dict(),dataset_digest=dataset_digest)
synchronize = lambda: synchronize_device(device_type)
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3: efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config_cls = LLWMConfig if args.model_type == "llwm" else GPTConfig
    extra = dict(predictor_layers=args.predictor_layers, action_dim=args.action_dim,
                 action_layers=args.action_layers, target_kl=args.target_kl,
                 kl_lambda=args.kl_lambda, ce_weight=args.ce_weight,
                 memory_weight=args.memory_weight, memory_loss=args.memory_loss) if args.model_type == "llwm" else {}
    config = config_cls(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern, **extra,
    )
    with torch.device("meta"):
        model_meta = create_model(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized
if device.type == "xla":
    import torch_xla.core.xla_model as xm
    xm.broadcast_master_param(model)

# New checkpoints use a run-specific destination. Existing checkpoints are untouched.
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else (f"llwm_d{args.depth}" if args.model_type == "llwm" else f"d{args.depth}")
checkpoint_dir = args.checkpoint_dir or os.path.join(base_dir,"base_checkpoints",output_dirname,args.run_id)
resuming = args.resume_latest or args.resume_from_step >= 0
meta_data = runtime_state = None

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

planning = planning_for_model(model, args)
orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = compile_model(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = (params_counts['transformer_matrices'] + params_counts['lm_head']
                      + params_counts.get('action_encoder', 0) + params_counts.get('predictor', 0))
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)


# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Set after restoring the checkpoint manifest. A resumed extension holds the
# original schedule's terminal LR/momentum/decay values for its added phase.
schedule_extension_start = None

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    if schedule_extension_start is not None and it >= schedule_extension_start:
        return args.final_lr_frac
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    if schedule_extension_start is not None and it >= schedule_extension_start:
        return 0.90
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    if schedule_extension_start is not None and it >= schedule_extension_start:
        return 0.0
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# Exact recovery validates the total schedule; stop-after-step only limits this attempt.
contract=dict(model_config=model_config_kwargs,world_size=ddp_world_size,
    total_batch_size=total_batch_size,device_batch_size=args.device_batch_size,sequence_len=args.max_seq_len,
    num_iterations=num_iterations,dtype=str(COMPUTE_DTYPE),dataset_digest=dataset_digest,
    optimizer={name:getattr(args,name) for name in ('embedding_lr','unembedding_lr','matrix_lr','scalar_lr','weight_decay')},
    scheduler={name:getattr(args,name) for name in ('warmup_steps','warmdown_ratio','final_lr_frac')})
cache=args.checkpoint_cache or str(Path('/tmp/llwm-checkpoint-cache')/args.run_id)
checkpoints=TrainingCheckpoints(checkpoint_dir,args.run_id,ddp_rank,ddp_world_size,device,contract,
    keep=args.checkpoint_keep,shared=args.checkpoint_shared,cache=cache,
    allow_horizon_extension=args.resume_extend_schedule)
checkpoints.verify_storage()
if resuming:
    resume_step=checkpoints.latest() if args.resume_latest else args.resume_from_step
    with diag.span('checkpoint_restore',checkpoint_step=resume_step):
        model_data,optimizer_data,meta_data,runtime_state,checkpoint_manifest=checkpoints.load(resume_step)
        orig_model.load_state_dict(model_data,strict=True,assign=False)
        optimizer.load_state_dict(optimizer_data)
        del model_data,optimizer_data
        if args.resume_extend_schedule:
            schedule_extension_start=int(checkpoint_manifest['contract']['num_iterations'])
            if schedule_extension_start!=meta_data['step'] or num_iterations<=schedule_extension_start:
                raise ValueError('--resume-extend-schedule requires a completed checkpoint and a larger --num-iterations')
            print0(f"Extending completed {schedule_extension_start}-step schedule to {num_iterations} steps; "
                   f"continuation holds LR multiplier at {args.final_lr_frac}, Muon momentum at 0.90, and weight decay at 0")
    if meta_data['step']>num_iterations:
        raise ValueError('Checkpoint is past the configured training budget')
    if args.stop_after_step>0 and args.stop_after_step<=meta_data['step']:
        raise ValueError('stop-after-step must be beyond the resumed optimizer step')
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,args.device_batch_size,args.max_seq_len,split='train',device=device,options=data_options,
    resume_state_dict=runtime_state['loader'] if runtime_state else None)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer,args.device_batch_size,args.max_seq_len,split='val',device=device,options=data_options)
x,y,dataloader_state_dict=next(train_loader)
if runtime_state:
    restore_rng(runtime_state['rng'],device)

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
grad_accum_steps = accumulation_steps(total_batch_size,args.device_batch_size,args.max_seq_len,ddp_world_size)
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

def save_training_checkpoint():
    snapshot=(dataloader_state_dict.snapshot() if hasattr(dataloader_state_dict,'snapshot')
              else train_loader.state_dict(replay_current=True))
    metadata=dict(step=step,val_bpb=val_bpb,model_config=model_config_kwargs,
        parameter_counts=param_counts,num_scaling_params=num_scaling_params,num_flops_per_token=num_flops_per_token,
        core_results=results,user_config=user_config,run_id=args.run_id,
        device_batch_size=args.device_batch_size,max_seq_len=args.max_seq_len,total_batch_size=total_batch_size,
        dataloader_state_dict=dict(dataloader_state_dict),
        loop_state=dict(min_val_bpb=min_val_bpb,smooth_train_loss=smooth_train_loss,total_training_time=total_training_time))
    with diag.span('checkpoint_save',checkpoint_step=step):
        checkpoints.save(step,orig_model.state_dict(),optimizer.state_dict(),metadata,
                         dict(loader=snapshot,rng=capture_rng(device)))


@contextmanager
def evaluation_phase(name):
    # Preserve training stochastic state independently of optional evaluation.
    rng=capture_rng(device)
    try:
        with diag.span(name):
            yield
    finally:
        restore_rng(rng,device)
        model.train()


# Go!
while True:
    last_step = step == num_iterations or (args.stop_after_step>0 and step>=args.stop_after_step)
    diag.set_context(step=step,microstep=None,**{k:dataloader_state_dict.get(k) for k in ('pq_idx','rg_idx','epoch','batch_number')})
    diag.event('step_begin',detail=True)
    results={}
    validation_due=args.eval_every>0 and (last_step or step%args.eval_every==0)
    core_due=args.core_metric_every>0 and (last_step or (step>0 and step%args.core_metric_every==0))
    sample_due=args.sample_every>0 and (last_step or (step>0 and step%args.sample_every==0))
    # A resumed checkpoint stores the optimizer step that was just completed.
    # Do not replay an evaluation at that step during startup: it can hold the
    # distributed workers before the first post-resume optimizer update.
    planning_due=(args.planning_every>0 and (last_step or (step>0 and step%args.planning_every==0))
                  and not (resuming and step==meta_data["step"]))
    # Commit recovery state BEFORE optional evaluation can fail.
    if last_step or (step>0 and ((args.save_every>0 and step%args.save_every==0) or validation_due or core_due or sample_due)):
        save_training_checkpoint()
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if validation_due:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with evaluation_phase('validation'), disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    if core_due:
        model.eval()
        with evaluation_phase('core_evaluation'), disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task,
                                    planning=planning, max_new_tokens=args.eval_max_new_tokens)
        if master_process:
            checkpoints.store.put_json(f'evaluation_{step:06d}.json',results)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            ("planned_core_metric" if planning is not None else "core_metric"): results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if sample_due and master_process:
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = PlanningEngine(orig_model, tokenizer, planning) if planning is not None else Engine(orig_model, tokenizer) # avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with evaluation_phase('sample_generation'), disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=args.eval_max_new_tokens, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    if planning_due and master_process:
        from nanochat.planning_eval import PlanningOptions, planned_completion
        options=planning or PlanningOptions(args.plan_goal,args.plan_iterations,args.plan_lr,args.plan_consistency_weight,args.plan_seed)
        prompt_ids=tokenizer(args.planning_prompt,prepend='<|bos|>')
        with evaluation_phase('planning_evaluation'):
            planned=planned_completion(orig_model,tokenizer,prompt_ids,args.eval_max_new_tokens,options)
            print0('Planning sample: '+tokenizer.decode(planned['tokens'][0].tolist()))

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    t0 = time.time()
    train_loss_sum = None
    loss_component_sums = None
    for micro_step in range(grad_accum_steps):
        diag.set_context(microstep=micro_step,**{k:dataloader_state_dict.get(k) for k in ('pq_idx','rg_idx','epoch','batch_number')})
        with diag.span('forward',detail=True):
            loss_components = model(x,y,return_loss_components=True) if args.model_type=='llwm' else None
            loss = loss_components['loss'] if loss_components is not None else model(x,y)
        train_loss = loss.detach() # for logging
        train_loss_sum = train_loss if train_loss_sum is None else train_loss_sum + train_loss
        if loss_components is not None:
            if loss_component_sums is None:
                loss_component_sums = {key: value.detach() for key, value in loss_components.items()}
            else:
                loss_component_sums = {key: loss_component_sums[key] + value.detach()
                                       for key, value in loss_components.items()}
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        with diag.span('backward',detail=True):
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        xla_microstep_boundary(device,args.xla_sync_mode)
        with diag.span('next_batch',detail=True):
            x,y,dataloader_state_dict=next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # Report optimizer-step means across every accumulated microstep, rather than
    # exposing only the final microbatch's loss decomposition.
    train_loss = train_loss_sum / grad_accum_steps
    if loss_component_sums is not None:
        loss_components = {key: value / grad_accum_steps for key, value in loss_component_sums.items()}
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        with diag.span("optimizer_update"):
            scaler.step(optimizer)
        scaler.update()
    else:
        with diag.span("optimizer_update"):
            optimizer.step()
    model.zero_grad(set_to_none=True)
    component_names = ('ce', 'memory_loss', 'kl', 'kl_total', 'kl_excess', 'kl_penalty')
    metric_tensors = [train_loss.detach().float()]
    if loss_components is not None:
        metric_tensors.extend(loss_components[name].detach().float() for name in component_names)
    metric_values = torch.stack(metric_tensors).cpu().tolist() # one compact device-to-host transfer
    train_loss_f = metric_values[0]
    train_loss_components = dict(zip(component_names, metric_values[1:])) if loss_components is not None else {}
    with diag.span('step_synchronize',detail=True):
        synchronize()
    if args.xla_metrics_every>0 and (step+1)%args.xla_metrics_every==0:
        diag.event('xla_metrics',metrics=diag.xla_metrics(device))
    diag.event('step_complete',completed_step=step+1,train_losses=dict(loss= train_loss_f, **train_loss_components))
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    loss_detail = " | losses: " + " ".join(f"{name}={value:.6f}" for name, value in train_loss_components.items()) if train_loss_components else ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {train_loss_f:.6f} | loss_ema: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}{loss_detail}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        if loss_components is not None:
            log_data.update({f"train/{key}": value.detach().item() for key, value in loss_components.items() if key != "loss"})
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == meta_data["step"])
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
