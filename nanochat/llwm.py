"""Latent-action language world model, adapted from LLWM_micro (without EQM).

The nanochat causal transformer compresses tokens into memories. A Gaussian
posterior encodes observed memory transitions; an action-conditioned predictor
and vocabulary head reconstruct the next token and the next memory. Prediction
without targets uses a fixed zero action (the standard normal prior's mean).
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.gpt import GPT, GPTConfig, Linear


@dataclass
class LLWMConfig(GPTConfig):
    architecture: str = "llwm"
    predictor_layers: int | None = None
    action_dim: int = 128
    action_layers: int = 2
    target_kl: float = 1.0
    kl_lambda: float = 1.0
    ce_weight: float = 1.0
    memory_weight: float = 1.0
    memory_loss: str = "smooth_l1"

    def __post_init__(self):
        if self.architecture != "llwm":
            raise ValueError("LLWMConfig architecture must be llwm")
        if self.sequence_len < 1 or self.n_layer < 1:
            raise ValueError("context length and compressor depth must be positive")
        if self.predictor_layers is None:
            self.predictor_layers = (3 * self.n_layer + 3) // 4
        if self.predictor_layers < 1 or self.action_layers < 1:
            raise ValueError("predictor and action encoder depths must be positive")
        if not 1 <= self.action_dim <= 2 * self.n_embd:
            raise ValueError("action_dim must be between 1 and twice the model width")
        if self.n_head < 1 or self.n_embd % self.n_head or (self.n_embd // self.n_head) % 2:
            raise ValueError("width and head count must give an even head dimension")
        if self.memory_loss not in ("mse", "smooth_l1"):
            raise ValueError("memory_loss must be mse or smooth_l1")
        for name in ("target_kl", "kl_lambda", "ce_weight", "memory_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


class ConditionLinear(nn.Linear):
    """BF16 activations with FP32 master weights, including biases."""
    def forward(self, x):
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.weight.to(x.dtype), bias)


class MemoryNorm(nn.LayerNorm):
    def forward(self, x):
        # Normalize in FP32, retaining the compute dtype on the residual path.
        return F.layer_norm(x.float(), self.normalized_shape,
                            self.weight, self.bias, self.eps).to(x.dtype)


class ActionEncoder(nn.Module):
    """Progressively halved MLP with separate Gaussian mean/log-std heads."""
    def __init__(self, width, action_dim=128, num_layers=2):
        super().__init__()
        if not 1 <= action_dim <= 2 * width or num_layers < 1:
            raise ValueError("invalid action dimension or action encoder depth")
        layers, current = [], 2 * width
        for layer in range(num_layers):
            following = action_dim if layer == num_layers - 1 else max(action_dim, current // 2)
            layers.extend((ConditionLinear(current, following), nn.GELU()))
            current = following
        self.trunk = nn.Sequential(*layers)
        self.mean = ConditionLinear(action_dim, action_dim)
        self.log_std = ConditionLinear(action_dim, action_dim)

    def forward(self, inputs):
        hidden = self.trunk(inputs)
        return torch.cat((self.mean(hidden), self.log_std(hidden)), dim=-1)


class AdaptiveBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = MemoryNorm(config.n_embd, elementwise_affine=False)
        self.condition = ConditionLinear(config.action_dim, 2 * config.n_embd)
        self.net = nn.Sequential(ConditionLinear(config.n_embd, 4 * config.n_embd),
                                 nn.GELU(), ConditionLinear(4 * config.n_embd, config.n_embd))

    def forward(self, memory, action):
        shift, scale = self.condition(action.to(memory.dtype)).chunk(2, dim=-1)
        return memory + self.net(self.norm(memory) * (1 + scale) + shift)


class LLWM(GPT):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to)
        self.compressor_norm = MemoryNorm(config.n_embd)
        self.action_encoder = ActionEncoder(config.n_embd, config.action_dim, config.action_layers)
        self.predictor = nn.ModuleList(AdaptiveBlock(config) for _ in range(config.predictor_layers))
        self.predictor_norm = MemoryNorm(config.n_embd)

    @property
    def compressor(self):
        return self.transformer.h

    @property
    def llm_head(self):
        return self.lm_head

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        # Constructors may run on meta; initialize every new parameter here.
        for module in (self.compressor_norm, self.action_encoder, self.predictor, self.predictor_norm):
            for layer in module.modules():
                if isinstance(layer, (nn.Linear, nn.LayerNorm)):
                    layer.reset_parameters()

    def encode(self, idx, kv_cache=None):
        return self.compressor_norm(super().encode(idx, kv_cache=kv_cache))

    def posterior(self, memory, next_memory, *, detach_inputs=True):
        if detach_inputs:
            memory, next_memory = memory.detach(), next_memory.detach()
        mean, log_std = self.action_encoder(torch.cat((memory, next_memory), dim=-1)).chunk(2, dim=-1)
        return mean.float(), log_std.float().clamp(-5, 2)

    def predict_features(self, memory, action):
        for block in self.predictor:
            memory = block(memory, action)
        return self.predictor_norm(memory)

    def decode_memory(self, memory):
        logits = self.llm_head(memory)[..., :self.config.vocab_size].float()
        return 15 * torch.tanh(logits / 15)

    def predict_logits(self, memory, action):
        return self.decode_memory(self.predict_features(memory, action))

    def prior_logits(self, memory):
        action = memory.new_zeros((*memory.shape[:-1], self.config.action_dim))
        return self.predict_logits(memory, action)

    def training_losses(self, idx, targets, memory=None):
        if idx.shape != targets.shape:
            raise ValueError("idx and targets must have matching [batch, sequence] shapes")
        memory = self.encode(idx) if memory is None else memory
        # Retain shifted compressor targets. The final transition needs one extra
        # observed token. Recompute its prefix without autograd so no training KV
        # cache mutation or autograd aliasing is needed on lazy XLA tensors.
        with torch.no_grad():
            full_idx = torch.cat((idx, targets[:, -1:].clamp(min=0)), dim=1)
            final_memory = self.encode(full_idx)[:, -1:]
            next_memory = torch.cat((memory[:, 1:].detach(), final_memory), dim=1)
        mean, log_std = self.posterior(memory, next_memory)
        action = mean + log_std.exp() * torch.randn_like(mean)
        predicted = self.predict_features(memory, action)
        logits = self.decode_memory(predicted)
        valid = targets.ne(-1)
        count = valid.sum().clamp(min=1)
        per_token_ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                       ignore_index=-1, reduction="none").view_as(targets)
        ce = per_token_ce.sum() / count
        regression = F.mse_loss if self.config.memory_loss == "mse" else F.smooth_l1_loss
        per_token_memory = regression(predicted.float(), next_memory.float(), reduction="none").mean(-1)
        memory_loss = (per_token_memory * valid).sum() / count
        per_dim_kl = 0.5 * (mean.square() + (2 * log_std).exp() - 1 - 2 * log_std)
        kl_total = (per_dim_kl * valid.unsqueeze(-1)).sum() / count
        kl = kl_total / self.config.action_dim
        kl_excess = F.relu(kl_total - self.config.target_kl)
        kl_penalty = self.config.kl_lambda * kl_excess.square()
        loss = self.config.ce_weight * ce + self.config.memory_weight * memory_loss + kl_penalty
        return dict(loss=loss, ce=ce, memory_loss=memory_loss, kl=kl,
                    kl_total=kl_total, kl_excess=kl_excess, kl_penalty=kl_penalty)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean", *, return_loss_components=False):
        if loss_reduction not in ("mean", "sum", "none"):
            raise ValueError("loss_reduction must be mean, sum or none")
        if targets is not None and targets.shape != idx.shape:
            raise ValueError("idx and targets must have matching shapes")
        if return_loss_components or (targets is not None and self.training and loss_reduction != "none"):
            if targets is None or kv_cache is not None:
                raise ValueError("world-model training requires targets and no inference KV cache")
            values = self.training_losses(idx, targets)
            if return_loss_components:
                return values
            return values["loss"] if loss_reduction == "mean" else values["loss"] * targets.ne(-1).sum()
        logits = self.prior_logits(self.encode(idx, kv_cache=kv_cache))
        if targets is None:
            return logits
        # Evaluation and RL log-probabilities must not see posterior/future tokens.
        per_token = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                    ignore_index=-1, reduction="none")
        if loss_reduction == "none":
            return per_token
        return per_token.sum() if loss_reduction == "sum" else per_token.sum() / targets.ne(-1).sum().clamp(min=1)

    def extra_optimizer_groups(self, matrix_lr, scalar_lr, weight_decay):
        groups = []
        # Posterior groups stay separate: RL uses the prior and skips these groups.
        for modules in ((self.action_encoder,), (self.compressor_norm, self.predictor, self.predictor_norm)):
            params = [p for module in modules for p in module.parameters()]
            for shape in sorted({p.shape for p in params if p.ndim == 2}):
                groups.append(dict(kind="muon", params=[p for p in params if p.shape == shape],
                                   lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay))
            others = [p for p in params if p.ndim != 2]
            if others:
                groups.append(dict(kind="adamw", params=others, lr=matrix_lr, betas=(0.9, 0.95),
                                   eps=1e-8, weight_decay=0.0))
        return groups

    def num_scaling_params(self):
        counts = dict(wte=sum(p.numel() for p in self.transformer.wte.parameters()),
                      value_embeds=sum(p.numel() for p in self.value_embeds.parameters()),
                      lm_head=sum(p.numel() for p in self.llm_head.parameters()),
                      transformer_matrices=sum(p.numel() for p in self.compressor.parameters()),
                      action_encoder=sum(p.numel() for p in self.action_encoder.parameters()),
                      predictor=sum(p.numel() for p in self.predictor.parameters()),
                      memory_norms=sum(p.numel() for m in (self.compressor_norm, self.predictor_norm) for p in m.parameters()))
        total = sum(p.numel() for p in self.parameters())
        counts['scalars'] = total - sum(counts.values())
        counts['total'] = total
        return counts

    def num_matmul_params(self):
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, (Linear, ConditionLinear)))

    def estimate_flops(self):
        # Approximate training work, including the detached extra encoder pass.
        base_matmuls = sum(m.weight.numel() for m in self.compressor.modules() if isinstance(m, Linear))
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attention = sum(12 * h * q * (t if w < 0 else min(w, t)) for w, _ in self.window_sizes)
        return 6 * self.num_matmul_params() + attention + 2 * base_matmuls + attention / 3

    def estimate_decode_flops(self, context_len):
        posterior = sum(m.weight.numel() for m in self.action_encoder.modules() if isinstance(m, ConditionLinear))
        return super().estimate_decode_flops(context_len) - 2 * posterior

    def estimate_prefill_flops(self, num_tokens):
        posterior = sum(m.weight.numel() for m in self.action_encoder.modules() if isinstance(m, ConditionLinear))
        return super().estimate_prefill_flops(num_tokens) - 2 * posterior * num_tokens
