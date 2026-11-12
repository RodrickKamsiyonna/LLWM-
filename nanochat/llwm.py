"""Latent-action language world model, adapted from LLWM_micro (without EQM).

The nanochat causal transformer compresses tokens into memories. A Gaussian
posterior encodes observed transitions into actions. An action-conditioned
predictor forecasts future memories. With head_source='action', the sampled
action is projected to model width and decoded through an MLP/vocabulary head.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.common import COMPUTE_DTYPE


@dataclass
class LLWMConfig(GPTConfig):
    architecture: str = "llwm"
    predictor_layers: int | None = None
    action_dim: int = 8
    action_layers: int = 2
    prediction_horizon: int = 4
    head_training: str = "detached"
    head_source: str = "compressor"
    action_conditioning: str = "memory_pair"
    planning_mode: str = "predictor"
    target_kl: float = 1.0
    kl_lambda: float = 1.0
    kl_mode: str = "target"
    ce_weight: float = 1.0
    ce_threshold: float = 0.0
    ce_power: int = 1
    memory_weight: float = 1.0
    memory_loss: str = "smooth_l1"
    token_objective: str = "ce"
    embedding_temperature: float = 1.0
    embedding_distance_ce: bool = False
    embedding_mse_weight: float = 0.0
    embedding_variance_weight: float = 0.0
    embedding_std_floor: float = 0.8
    detach_embedding_target: bool = True
    head_layers: int = 2
    sigreg_weight: float = 0.0
    sigreg_projections: int = 1024
    sigreg_knots: int = 17

    def __post_init__(self):
        if not math.isfinite(self.embedding_temperature) or self.embedding_temperature <= 0:
            raise ValueError("embedding_temperature must be finite and positive")
        if not math.isfinite(self.embedding_mse_weight) or self.embedding_mse_weight < 0:
            raise ValueError("embedding_mse_weight must be finite and nonnegative")
        if not math.isfinite(self.embedding_variance_weight) or self.embedding_variance_weight < 0:
            raise ValueError("embedding_variance_weight must be finite and nonnegative")
        if not math.isfinite(self.embedding_std_floor) or self.embedding_std_floor <= 0:
            raise ValueError("embedding_std_floor must be finite and positive")
        if self.token_objective not in ("ce", "embedding_mse"):
            raise ValueError("invalid token objective")
        if self.token_objective == "embedding_mse" and self.action_conditioning != "next_token":
            raise ValueError("embedding MSE requires next_token action conditioning")
        if self.head_layers < 1 or self.sigreg_projections < 1 or self.sigreg_knots < 2:
            raise ValueError("invalid head/SIGReg configuration")
        if not math.isfinite(self.sigreg_weight) or self.sigreg_weight < 0:
            raise ValueError("sigreg_weight must be finite and nonnegative")
        if self.architecture != "llwm":
            raise ValueError("LLWMConfig architecture must be llwm")
        if self.sequence_len < 1 or self.n_layer < 1:
            raise ValueError("context length and compressor depth must be positive")
        if self.predictor_layers is None:
            self.predictor_layers = (3 * self.n_layer + 3) // 4
        if self.predictor_layers < 1 or self.action_layers < 1:
            raise ValueError("predictor and action encoder depths must be positive")
        if not isinstance(self.prediction_horizon, int) or self.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be a positive integer")
        if self.head_training not in ("joint", "detached"):
            raise ValueError("head_training must be joint or detached")
        if self.head_source not in ("compressor", "predictor", "action"):
            raise ValueError("head_source must be compressor, predictor or action")
        if self.action_conditioning not in ('memory_pair', 'next_token'):
            raise ValueError('action_conditioning must be memory_pair or next_token')
        if self.planning_mode not in ('predictor', 'straight_through'):
            raise ValueError('planning_mode must be predictor or straight_through')
        if self.action_conditioning == 'next_token':
            if self.head_source != 'predictor' or self.memory_weight != 0:
                raise ValueError('next_token conditioning requires predictor head and memory_weight=0')
            if self.prediction_horizon != 1:
                raise ValueError('next_token training uses one-step teacher forcing; set prediction_horizon=1')
        if self.planning_mode == 'straight_through' and self.head_source != 'predictor':
            raise ValueError('straight-through planning requires a predictor token head')
        if self.ce_power not in (1, 2):
            raise ValueError("ce_power must be 1 or 2")
        if self.kl_mode not in ("target", "standard"):
            raise ValueError("kl_mode must be target or standard")
        if not 1 <= self.action_dim <= 2 * self.n_embd:
            raise ValueError("action_dim must be between 1 and twice the model width")
        if self.n_head < 1 or self.n_embd % self.n_head or (self.n_embd // self.n_head) % 2:
            raise ValueError("width and head count must give an even head dimension")
        if self.memory_loss not in ("mse", "smooth_l1"):
            raise ValueError("memory_loss must be mse or smooth_l1")
        for name in ("target_kl", "kl_lambda", "ce_weight", "ce_threshold", "memory_weight"):
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
    def __init__(self, width, action_dim=8, num_layers=2):
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
        head_modules = []
        for index in range(config.head_layers):
            head_modules.append(ConditionLinear(config.n_embd, config.n_embd))
            if index < config.head_layers - 1:
                head_modules.append(nn.GELU())
        self.llm_head_mlp = nn.Sequential(*head_modules)
        if config.token_objective == "embedding_mse":
            self.embedding_head = ConditionLinear(config.n_embd, config.n_embd)
        if config.head_source == "action":
            self.action_projection = ConditionLinear(config.action_dim, config.n_embd)

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
        modules = [self.compressor_norm, self.action_encoder, self.predictor,
                   self.predictor_norm, self.llm_head_mlp]
        if hasattr(self, 'embedding_head'):
            modules.append(self.embedding_head)
        if hasattr(self, 'action_projection'):
            modules.append(self.action_projection)
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, (nn.Linear, nn.LayerNorm)):
                    layer.reset_parameters()

    def encode(self, idx, kv_cache=None):
        return self.compressor_norm(super().encode(idx, kv_cache=kv_cache))

    def encode_st_tokens(self, probabilities, token_ids, kv_cache=None):
        return self.compressor_norm(super().encode_st_tokens(probabilities, token_ids, kv_cache))

    def encode_soft_tokens(self, probabilities, token_ids, kv_cache=None):
        return self.compressor_norm(super().encode_soft_tokens(probabilities, token_ids, kv_cache))

    def posterior(self, memory, next_memory, *, detach_inputs=True):
        if detach_inputs:
            memory, next_memory = memory.detach(), next_memory.detach()
        mean, log_std = self.action_encoder(torch.cat((memory, next_memory), dim=-1)).chunk(2, dim=-1)
        return mean.float(), log_std.float().clamp(-5, 2)

    def token_posterior(self, memory, next_tokens):
        """q(action | current latent, observed next token), with joint gradients."""
        target_embeddings = self.transformer.wte(next_tokens.clamp(min=0))
        # Match LLWM_micro: the observed next-token embedding is a detached
        # teacher in BOTH the regression target and the posterior input.
        # Stop posterior gradients into both observed inputs; the predictor's
        # separate memory input still trains the causal compressor.
        if self.config.token_objective == 'embedding_mse':
            memory = memory.detach()
            target_embeddings = target_embeddings.detach()
        target_embeddings = target_embeddings.to(memory.dtype)
        mean, log_std = self.action_encoder(torch.cat((memory, target_embeddings), dim=-1)).chunk(2, dim=-1)
        return mean.float(), log_std.float().clamp(-5, 2)

    def predict_features(self, memory, action):
        for block in self.predictor:
            memory = block(memory, action)
        return self.predictor_norm(memory)

    def decode_memory(self, memory, *, detach=False):
        head_input = memory.detach() if detach else memory
        hidden = self.llm_head_mlp(head_input)
        logits = self.llm_head(hidden)[..., :self.config.vocab_size].float()
        return 15 * torch.tanh(logits / 15)

    def predict_embedding(self, memory, action):
        return self.embedding_head(self.llm_head_mlp(self.predict_features(memory, action))).float()

    def embedding_logits(self, embedding):
        # Negative squared Euclidean distance: decoding matches the MSE target space.
        table = self.transformer.wte.weight[:self.config.vocab_size].float()
        return (2 * embedding.float() @ table.T - table.square().sum(-1)) / self.config.n_embd

    def predict_planning_output(self, memory, action):
        if self.config.token_objective == 'embedding_mse':
            return self.predict_embedding(memory, action)
        return self.predict_logits(memory, action)

    def encode_predicted_embedding(self, embedding, kv_cache):
        ids = self.embedding_logits(embedding).argmax(-1, keepdim=True)
        # Main compressor input is the continuous predicted embedding. Token IDs
        # select auxiliary value embeddings only; the primary gradient stays continuous.
        return self.compressor_norm(super().encode_embeddings(
            embedding[:, None].to(COMPUTE_DTYPE), ids, kv_cache=kv_cache,
            token_probabilities=self.embedding_logits(embedding).softmax(-1)[:, None], soft_tokens=True))[:, -1]

    def predict_logits(self, memory, action):
        if self.config.token_objective == "embedding_mse":
            return self.embedding_logits(self.predict_embedding(memory, action))
        if self.config.head_source == "action":
            return self.decode_actions(action)
        return self.decode_memory(self.predict_features(memory, action))

    def decode_actions(self, action, *, detach=False):
        """Decode the same sampled/planned action that drives the predictor."""
        if self.config.head_source != "action":
            raise ValueError('action decoding requires head_source=action')
        head_input = action.detach() if detach else action
        projected = self.action_projection(head_input.to(COMPUTE_DTYPE))
        return self.decode_memory(projected)

    def prior_logits(self, memory):
        """Causal inference uses the zero-action prior for predictor heads."""
        if self.config.head_source in ("predictor", "action"):
            action = memory.new_zeros((*memory.shape[:-1], self.config.action_dim))
            return self.predict_logits(memory, action)
        return self.decode_memory(memory)

    def training_losses(self, idx, targets, memory=None):
        if idx.shape != targets.shape:
            raise ValueError("idx and targets must have matching [batch, sequence] shapes")
        memory = self.encode(idx) if memory is None else memory
        if self.config.action_conditioning == 'next_token':
            mean, log_std = self.token_posterior(memory, targets)
            action = mean + log_std.exp() * torch.randn_like(mean)
            if self.config.token_objective == 'embedding_mse':
                predicted_embedding = self.predict_embedding(memory, action)
                target_embedding = self.transformer.wte(targets.clamp(min=0))
                if self.config.detach_embedding_target:
                    target_embedding = target_embedding.detach()
                target_embedding = target_embedding.float()
                valid = targets.ne(-1)
                mse = ((predicted_embedding - target_embedding).square().mean(-1) * valid).sum() / valid.sum().clamp(min=1)
                components = self._loss_components(mse.new_zeros(()), mse.new_zeros(()), mean, log_std, valid)
                components['embedding_mse'] = mse
                # Across vocabulary rows: a constant embedding table has zero variance.
                # FP32 moments avoid BF16 cancellation; no vocabulary covariance matrix.
                table = self.transformer.wte.weight.float()
                if self.config.embedding_variance_weight:
                    variance = (table.square().mean(0) - table.mean(0).square()).clamp(min=0)
                    std = (variance + 1e-4).sqrt()
                    anti_collapse = F.relu(self.config.embedding_std_floor - std).square().mean()
                else:
                    anti_collapse = mse.new_zeros(())
                components['embedding_variance'] = anti_collapse
                components['embedding_variance_penalty'] = self.config.embedding_variance_weight * anti_collapse
                components['embedding_rms'] = table.square().mean().sqrt().detach()
                if self.config.embedding_distance_ce:
                    # Softmax(-MSE / temperature). The common -||prediction||^2/D
                    # cancels in softmax. All vocabulary embeddings receive gradients.
                    logits = self.embedding_logits(predicted_embedding) / self.config.embedding_temperature
                    distance_ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                                  ignore_index=-1, reduction='sum') / valid.sum().clamp(min=1)
                    components['ce'] = distance_ce
                    components['loss'] = self.config.ce_weight * distance_ce + self.config.embedding_mse_weight * mse + components['kl_penalty']
                else:
                    components['loss'] = mse + components['kl_penalty'] + components['embedding_variance_penalty']
                return components
            predicted = self.predict_features(memory, action)
            logits = self.decode_memory(predicted, detach=self.config.head_training == 'detached')
            ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                 ignore_index=-1, reduction='sum') / targets.ne(-1).sum().clamp(min=1)
            return self._loss_components(ce, memory.new_zeros((), dtype=torch.float32),
                                         mean, log_std, targets.ne(-1))
        # Retain shifted compressor targets. The final transition needs one extra
        # observed token. Recompute its prefix without autograd so no training KV
        # cache mutation or autograd aliasing is needed on lazy XLA tensors.
        with torch.no_grad():
            full_idx = torch.cat((idx, targets[:, -1:].clamp(min=0)), dim=1)
            final_memory = self.encode(full_idx)[:, -1:]
            next_memory = torch.cat((memory[:, 1:].detach(), final_memory), dim=1)
        mean, log_std = self.posterior(memory, next_memory)
        action = mean + log_std.exp() * torch.randn_like(mean)
        valid = targets.ne(-1)
        regression = F.mse_loss if self.config.memory_loss == "mse" else F.smooth_l1_loss

        # Preserve prior head paths for older checkpoints. The action head uses
        # the same sampled action for next-token CE and memory prediction.
        head_memory = memory if self.config.head_source == "compressor" else None

        memory_sum = memory.new_zeros((), dtype=torch.float32)
        prediction_count = memory.new_zeros((), dtype=torch.float32)
        predicted = memory
        for offset in range(1, min(self.config.prediction_horizon, memory.size(1)) + 1):
            count = memory.size(1) - offset + 1
            predicted = self.predict_features(predicted[:, :count], action[:, offset - 1:offset - 1 + count])
            if offset == 1 and self.config.head_source == "predictor":
                head_memory = predicted
            destination = next_memory[:, offset - 1:offset - 1 + count]
            mask = valid[:, :count]
            for intermediate in range(1, offset):
                mask = mask & valid[:, intermediate:intermediate + count]
            per_token_memory = regression(predicted.float(), destination.float(), reduction="none").mean(-1)
            memory_sum = memory_sum + (per_token_memory * mask).sum()
            prediction_count = prediction_count + mask.sum()
        count = prediction_count.clamp(min=1)
        memory_loss = memory_sum / count
        if self.config.head_source == "action":
            logits = self.decode_actions(action, detach=self.config.head_training == "detached")
        else:
            logits = self.decode_memory(head_memory, detach=self.config.head_training == "detached")
        ce_targets = idx if self.config.head_source == "compressor" else targets
        ce_sum = F.cross_entropy(logits.flatten(0, 1), ce_targets.flatten(),
                                ignore_index=-1, reduction="sum")
        ce = ce_sum / ce_targets.ne(-1).sum().clamp(min=1)
        components = self._loss_components(ce, memory_loss, mean, log_std, valid)
        if self.config.sigreg_weight:
            from nanochat.sigreg import sigreg_loss
            regularizer = sigreg_loss(memory, valid, self.config.sigreg_projections, self.config.sigreg_knots)
        else:
            regularizer = memory.new_zeros((), dtype=torch.float32)
        components['sigreg'] = regularizer
        components['sigreg_penalty'] = self.config.sigreg_weight * regularizer
        components['loss'] = components['loss'] + components['sigreg_penalty']
        return components

    def _loss_components(self, ce, memory_loss, mean, log_std, valid):
        action_count = valid.sum().clamp(min=1)
        per_dim_kl = 0.5 * (mean.square() + (2 * log_std).exp() - 1 - 2 * log_std)
        kl_total = (per_dim_kl * valid.unsqueeze(-1)).sum() / action_count
        kl = kl_total / self.config.action_dim
        if self.config.kl_mode == "standard":
            # Sum KL over action dimensions, average over valid transitions.
            # No free budget or squared excess: penalize KL even below 1 nat.
            kl_excess = torch.zeros_like(kl_total)
            kl_penalty = self.config.kl_lambda * kl_total
        else:
            kl_excess = F.relu(kl_total - self.config.target_kl)
            kl_penalty = self.config.kl_lambda * kl_excess.square()
        # Threshold the mean CE for this rank/microbatch, before accumulation.
        # Below the threshold the token head contributes no CE gradient.
        ce_excess = F.relu(ce - self.config.ce_threshold)
        ce_penalty = ce_excess.square() if self.config.ce_power == 2 else ce_excess
        loss = self.config.ce_weight * ce_penalty + self.config.memory_weight * memory_loss + kl_penalty
        return dict(embedding_variance=loss.new_zeros(()), embedding_variance_penalty=loss.new_zeros(()), embedding_rms=loss.new_zeros(()), embedding_mse=loss.new_zeros(()), sigreg=loss.new_zeros(()), sigreg_penalty=loss.new_zeros(()), loss=loss, ce=ce, ce_excess=ce_excess, ce_penalty=ce_penalty, memory_loss=memory_loss, kl=kl,
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
                      lm_head=(sum(p.numel() for p in self.embedding_head.parameters()) if hasattr(self, 'embedding_head') else 0) +
                               sum(p.numel() for p in self.llm_head.parameters()) +
                               sum(p.numel() for p in self.llm_head_mlp.parameters()) +
                               (sum(p.numel() for p in self.action_projection.parameters())
                                if hasattr(self, 'action_projection') else 0),
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
        if self.config.action_conditioning == 'next_token':
            return 6 * self.num_matmul_params() + attention
        return 6 * self.num_matmul_params() + attention + 2 * base_matmuls + attention / 3

    def estimate_decode_flops(self, context_len):
        posterior = sum(m.weight.numel() for m in self.action_encoder.modules() if isinstance(m, ConditionLinear))
        return super().estimate_decode_flops(context_len) - 2 * posterior

    def estimate_prefill_flops(self, num_tokens):
        posterior = sum(m.weight.numel() for m in self.action_encoder.modules() if isinstance(m, ConditionLinear))
        return super().estimate_prefill_flops(num_tokens) - 2 * posterior * num_tokens
