"""Optimize actions through autoregressive predicted-memory transitions."""
import math
from nanochat.common import synchronize_device
import torch
from torch.nn import functional as F


class DifferentiableKVCache:
    """Functional per-layer KV append cache for gradient-based soft-token rollouts."""
    differentiable = True

    def __init__(self, prefix_cache):
        self.position = prefix_cache.get_pos()
        self.max_seq_len = prefix_cache.max_seq_len
        self.n_layers = prefix_cache.n_layers
        self.k_layers = [prefix_cache.k_cache[i, :, :self.position] for i in range(self.n_layers)]
        self.v_layers = [prefix_cache.v_cache[i, :, :self.position] for i in range(self.n_layers)]
        self.prev_embedding = prefix_cache.prev_embedding

    def get_pos(self):
        return self.position

    def append_layer(self, layer, key, value):
        if self.position >= self.max_seq_len:
            raise ValueError('soft-token rollout exceeds its KV cache allocation')
        self.k_layers[layer] = torch.cat((self.k_layers[layer], key), dim=1)
        self.v_layers[layer] = torch.cat((self.v_layers[layer], value), dim=1)
        return self.k_layers[layer], self.v_layers[layer]

    def advance(self, count):
        self.position += count


def soft_compressor_rollout(model, initial_memory, prefix_cache, actions, temperature=1.0):
    """Predict tokens, feed their expected embeddings through the cached compressor."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('soft-token temperature must be finite and positive')
    cache = DifferentiableKVCache(prefix_cache)
    memory = initial_memory
    logits = []
    embedding_table = model.transformer.wte.weight[:model.config.vocab_size].float()
    for index, action in enumerate(actions.unbind(1)):
        predicted_memory = model.predict_features(memory, action[:, None, :])
        step_logits = model.decode_memory(predicted_memory)
        logits.append(step_logits)
        if index + 1 < actions.size(1):
            probabilities = F.softmax(step_logits.float() / temperature, dim=-1)
            soft_embedding = torch.matmul(probabilities, embedding_table).to(model.transformer.wte.weight.dtype)
            memory = model.encode_soft_embeddings(soft_embedding, probabilities, kv_cache=cache)
    return torch.cat(logits, dim=1)


def plan_soft_compressor(model, prompt, goal, horizon=32, iterations=100, lr=0.05,
                         seed=123, temperature=1.0, return_logits=False):
    """Optimize latent actions through predictor, LM head and soft-token compressor recurrences."""
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('planning lr must be finite and positive')
    if prompt.ndim != 2 or prompt.size(1) < 1:
        raise ValueError('prompt must be [batch, nonempty sequence]')
    if goal.ndim != 1 or goal.numel() < 1 or horizon < goal.numel() or iterations < 1:
        raise ValueError('goal must be nonempty; horizon >= goal length; iterations >= 1')
    if horizon >= model.config.sequence_len:
        raise ValueError('planning horizon must leave at least one context position for the prompt')

    from nanochat.engine import KVCache
    training = model.training
    flags = [p.requires_grad for p in model.parameters()]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        context = model.config.sequence_len
        prompt = prompt[:, -(min(prompt.size(1), context-horizon)):]
        if prompt.size(1) < 1:
            raise ValueError('no prompt tokens remain after reserving planning horizon')
        cache = KVCache(batch_size=prompt.size(0), num_heads=model.config.n_kv_head,
                        seq_len=prompt.size(1)+horizon,
                        head_dim=model.config.n_embd//model.config.n_head,
                        num_layers=model.config.n_layer, device=prompt.device,
                        dtype=model.transformer.wte.weight.dtype)
        with torch.no_grad():
            memories = model.encode(prompt, kv_cache=cache)
            initial = memories[:, -1:].detach()

        actions = torch.nn.Parameter(sample_actions(prompt.size(0), horizon, model.config.action_dim,
                                                    device=prompt.device, seed=seed))
        targets = goal[None].expand(prompt.size(0), -1)
        optimizer = torch.optim.Adam([actions], lr=lr)
        history, best_loss, best_actions = [], float('inf'), None
        best_goal_ce = float('inf')
        for iteration in range(iterations+1):
            with torch.set_grad_enabled(iteration < iterations):
                logits = soft_compressor_rollout(model, initial, cache, actions, temperature)
                goal_logits = logits[:, horizon-goal.numel():]
                goal_ce = F.cross_entropy(goal_logits.flatten(0, 1), targets.flatten())
            value = float(goal_ce.detach())
            if not math.isfinite(value):
                raise RuntimeError('non-finite planning objective')
            exact = goal_logits.argmax(-1).eq(targets).all(-1).float().mean()
            history.append({'iteration': iteration, 'loss': value, 'goal_ce': value,
                            'goal_exact_match': float(exact.detach())})
            if value < best_loss:
                best_loss = value
                best_goal_ce = value
                best_actions = actions.detach().clone()
            if iteration < iterations:
                optimizer.zero_grad(set_to_none=True)
                goal_ce.backward()
                torch.nn.utils.clip_grad_norm_([actions], 1.0)
                optimizer.step()
                synchronize_device(prompt.device.type)

        with torch.no_grad():
            best_logits = soft_compressor_rollout(model, initial, cache, best_actions, temperature)
            tokens = best_logits.argmax(-1)
            suffix = tokens[:, horizon-goal.numel():]
        result = {'tokens': tokens.cpu(), 'goal_tokens': suffix.cpu(),
                  'goal_exact_match': suffix.eq(targets).all(-1).float().mean().item(),
                  'history': history, 'best_loss': best_loss, 'goal_ce': best_goal_ce,
                  'temperature': temperature}
        if return_logits:
            result['logits'] = best_logits.detach().cpu()
        return result
    finally:
        for parameter, flag in zip(model.parameters(), flags):
            parameter.requires_grad_(flag)
        model.train(training)


def rollout(model, memory, actions, *, return_consistency=False):
    """Recur through memories; optionally infer each transition's own mean action."""
    memories, previous = [], []
    for action in actions.unbind(1):
        previous.append(memory)
        memory = model.predict_features(memory, action)
        memories.append(memory)
    states = torch.stack(memories, 1)
    logits = model.decode_memory(states)
    if not return_consistency:
        return logits
    mean, _ = model.posterior(torch.stack(previous, 1), states, detach_inputs=False)
    consistency = F.mse_loss(actions.float(), mean.float())
    return logits, consistency


def plan(model, prompt, goal, horizon=500, iterations=100, lr=0.05, seed=123, consistency_weight=1., return_logits=False):
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('planning lr must be finite and positive')
    if not math.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError('consistency_weight must be finite and nonnegative')
    if goal.ndim != 1 or goal.numel() < 1 or horizon < goal.numel() or iterations < 1:
        raise ValueError('goal must be nonempty; horizon >= goal length; iterations >= 1')
    if prompt.ndim != 2 or prompt.size(1) < 1:
        raise ValueError('prompt must be [batch, nonempty sequence]')
    training = model.training
    flags = [p.requires_grad for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        with torch.no_grad():
            start=max(0,prompt.size(1)-model.config.sequence_len)
            states = model.encode(prompt[:, start:])
            initial = states[:, states.size(1)-1].detach()
        actions = torch.nn.Parameter(sample_actions(initial.size(0), horizon, model.config.action_dim,
                                                    device=initial.device, seed=seed))
        targets = goal[None].expand(initial.size(0), -1)
        optimizer = torch.optim.Adam([actions], lr=lr)
        history, best_loss, best_actions = [], float('inf'), None
        for i in range(iterations+1):
            with torch.set_grad_enabled(i < iterations):
                if consistency_weight:
                    logits, consistency = rollout(model, initial, actions, return_consistency=True)
                else:
                    logits = rollout(model, initial, actions)
                    consistency = logits.new_zeros(())
                goal_ce = F.cross_entropy(logits[:, horizon-goal.numel():].flatten(0, 1), targets.flatten())
                loss = goal_ce + consistency_weight*consistency
            value = float(loss.detach())
            if not math.isfinite(value):
                raise RuntimeError('non-finite planning objective')
            exact = logits[:, horizon-goal.numel():].argmax(-1).eq(targets).all(-1).float().mean()
            history.append({'iteration': i, 'loss': value, 'goal_ce': float(goal_ce.detach()), 'inverse_consistency': float(consistency.detach()),
                            'consistency_penalty': float(consistency_weight*consistency.detach()), 'goal_exact_match': float(exact)})
            if value < best_loss:
                best_loss, best_actions = value, actions.detach().clone()
                best_goal_ce, best_consistency = float(goal_ce.detach()), float(consistency.detach())
            if i < iterations:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([actions], 1.)
                optimizer.step()
                synchronize_device(initial.device.type)
        with torch.no_grad():
            logits = rollout(model, initial, best_actions)
            tokens = logits.argmax(-1)
            suffix = tokens[:, horizon-goal.numel():]
        result = {'tokens': tokens.cpu(), 'goal_tokens': suffix.cpu(), 'goal_exact_match': suffix.eq(targets).all(-1).float().mean().item(),
                'history': history, 'best_loss': best_loss, 'goal_ce': best_goal_ce,
                'inverse_consistency': best_consistency, 'consistency_penalty': consistency_weight*best_consistency}
        if return_logits:
            result['logits'] = logits.detach().cpu()
        return result
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
        model.train(training)


def sample_actions(batch_size, horizon, action_dim, *, device, seed=123):
    """Independent standard-normal initialization at every rollout step."""
    device = torch.device(device)
    rng_device = "cpu" if device.type == "xla" else device
    generator = torch.Generator(device=rng_device).manual_seed(seed)
    return torch.randn(batch_size, horizon, action_dim, generator=generator,
                       device=rng_device).to(device)
