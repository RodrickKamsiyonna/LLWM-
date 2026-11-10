"""Goal-probability MPC through hard token forwards and soft embedding gradients."""
import math
import time

import torch
from torch.nn import functional as F

from nanochat import diagnostics as diag
from nanochat.common import print0
from nanochat.engine import DifferentiableKVCache
from nanochat.planning import _freeze_model, _restore_model, _trim_prompt, sample_actions


def _dispatch(device, *, wait=False):
    if device.type == 'xla':
        import torch_xla
        torch_xla.sync(wait=wait)


def _st_step(model, memory, action, cache):
    logits = model.predict_planning_output(memory, action)
    if model.config.token_objective == "embedding_mse":
        next_h = model.encode_predicted_embedding(logits, cache)
        return (logits, next_h, *cache.tensors())
    selected = logits.argmax(-1, keepdim=True)
    probabilities = logits.softmax(-1).unsqueeze(1)
    encoder = model.encode_soft_tokens if getattr(model, "planning_embedding_mode", "soft") == "soft" else model.encode_st_tokens
    next_h = encoder(probabilities, selected, kv_cache=cache)[:, -1]
    return (logits, next_h, *cache.tensors())


class _StagewiseSTRollout(torch.autograd.Function):
    """Exact local VJPs bound the XLA backward graph to one token at a time.

    Reentrant checkpoints previously joined all cached-state adjoints into one
    enormous XLA graph. Here forward saves detached states, and backward passes
    both the latent and *all* KV/smear adjoints through each recomputed step.
    This preserves the straight-through gradient, including earlier actions.
    """
    @staticmethod
    def forward(ctx, actions, initial, model, prefix_cache):
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('stagewise planning requires frozen model parameters')
        cache, memory, outputs, saved = prefix_cache.fork(), initial, [], []
        for index, action in enumerate(actions.unbind(1)):
            saved.extend((memory, *cache.tensors()))
            if index == actions.size(1) - 1:
                logits = model.predict_planning_output(memory, action)
            else:
                position = cache.get_pos()
                result = _st_step(model, memory, action, cache)
                logits, memory = result[:2]
                cache = DifferentiableKVCache.from_tensors(result[2:], cache.n_layers,
                                                           cache.max_seq_len, position + 1)
            outputs.append(logits)
            _dispatch(actions.device)
        ctx.model = model
        ctx.layers, ctx.capacity, ctx.position = (prefix_cache.n_layers,
                                                 prefix_cache.max_seq_len, prefix_cache.get_pos())
        ctx.save_for_backward(actions, *saved)
        return torch.stack(outputs, dim=1)

    @staticmethod
    def backward(ctx, output_grad):
        actions, *saved = ctx.saved_tensors
        stride = 2 + 2 * ctx.layers  # latent, smear embedding, keys, values
        action_grads = [None] * actions.size(1)
        memory_grad, cache_grads = None, None
        _dispatch(actions.device, wait=True)
        for index in reversed(range(actions.size(1))):
            state = saved[index * stride:(index + 1) * stride]
            with torch.enable_grad():
                memory = state[0].detach().requires_grad_()
                action = actions[:, index].detach().requires_grad_()
                if index == actions.size(1) - 1:
                    logits = ctx.model.predict_planning_output(memory, action)
                    memory_grad, action_grad = torch.autograd.grad(
                        logits, (memory, action), output_grad[:, index])
                    cache_grads = tuple(torch.zeros_like(t) for t in state[1:])
                else:
                    cache_inputs = tuple(t.detach().requires_grad_() for t in state[1:])
                    cache = DifferentiableKVCache.from_tensors(cache_inputs, ctx.layers,
                                                               ctx.capacity, ctx.position + index)
                    outputs = _st_step(ctx.model, memory, action, cache)
                    inputs = (memory, action, *cache_inputs)
                    grads = torch.autograd.grad(outputs, inputs,
                                                (output_grad[:, index], memory_grad, *cache_grads),
                                                allow_unused=True)
                    grads = tuple(torch.zeros_like(t) if g is None else g for t,g in zip(inputs, grads))
                    memory_grad, action_grad, cache_grads = grads[0], grads[1], grads[2:]
            action_grads[index] = action_grad
            # Materialize each local VJP so its graph cannot grow across time.
            _dispatch(actions.device, wait=True)
        return torch.stack(action_grads, dim=1), memory_grad, None, None


def st_rollout(model, initial, prefix_cache, actions, *, checkpoint_steps=True, stagewise=None):
    """Recompress each selected hard token; preserve gradients through the KV history.

    The prefix is immutable. Rematerialized steps reconstruct their own cache
    object from tensor inputs, so backward never appends to a shared live cache.
    """
    if actions.ndim != 3 or actions.size(1) < 1 or initial.ndim != 2:
        raise ValueError('expected initial [batch,width] and nonempty [batch,horizon,action_dim]')
    if actions.size(0) != initial.size(0) or actions.size(2) != model.config.action_dim:
        raise ValueError('planning actions do not match model/batch dimensions')
    cache, memory, outputs = prefix_cache.fork(), initial, []
    differentiating = torch.is_grad_enabled() and actions.requires_grad
    stagewise = actions.device.type == 'xla' if stagewise is None else stagewise
    if stagewise and differentiating:
        return _StagewiseSTRollout.apply(actions, initial, model, prefix_cache)
    use_checkpoint = checkpoint_steps and differentiating
    if use_checkpoint:
        if actions.device.type == 'xla':
            from torch_xla.utils.checkpoint import checkpoint
            checkpoint_kwargs = dict(use_reentrant=True)
        else:
            from torch.utils.checkpoint import checkpoint
            checkpoint_kwargs = dict(use_reentrant=False)

    for index, action in enumerate(actions.unbind(1)):
        if index == actions.size(1) - 1:
            logits = model.predict_planning_output(memory, action)
        else:
            position = cache.get_pos()
            layers, capacity = cache.n_layers, cache.max_seq_len

            def step_fn(h, a, *cache_tensors, position=position, layers=layers, capacity=capacity):
                local = DifferentiableKVCache.from_tensors(cache_tensors, layers, capacity, position)
                return _st_step(model, h, a, local)

            args = (memory, action, *cache.tensors())
            result = (checkpoint(step_fn, *args, **checkpoint_kwargs)
                      if use_checkpoint else step_fn(*args))
            logits, memory = result[:2]
            cache = DifferentiableKVCache.from_tensors(result[2:], layers, capacity, position + 1)
        outputs.append(logits)
        if actions.device.type == 'xla':
            import torch_xla
            torch_xla.sync(wait=False)
    return torch.stack(outputs, dim=1)


def optimize_st_actions(model, initial, prefix_cache, goal, horizon, iterations, lr, seed,
                        *, action_init=None, round_index=0):
    actions = sample_actions(initial.size(0), horizon, model.config.action_dim,
                             device=initial.device, seed=seed) if action_init is None else action_init
    if actions.shape != (initial.size(0), horizon, model.config.action_dim):
        raise ValueError('warm-start actions do not match the remaining horizon')
    actions = torch.nn.Parameter(actions.detach().clone())
    optimizer = torch.optim.Adam([actions], lr=lr, capturable=actions.device.type == 'xla')
    targets = goal[None].expand(initial.size(0), -1)
    metric = 'goal_mse' if model.config.token_objective == 'embedding_mse' else 'goal_ce'
    best_loss, best_actions, history = float('inf'), None, []
    for iteration in range(iterations + 1):
        started = time.monotonic()
        diag.event('planning_iteration_begin', round=round_index, iteration=iteration,
                   iterations=iterations, horizon=horizon, mode='straight_through')
        with torch.set_grad_enabled(iteration < iterations):
            logits = st_rollout(model, initial, prefix_cache, actions)
            if model.config.token_objective == 'embedding_mse':
                target_embeddings = model.transformer.wte(targets).detach().float()
                loss = F.mse_loss(logits[:, -goal.numel():].float(), target_embeddings)
            else:
                loss = F.cross_entropy(logits[:, -goal.numel():].flatten(0, 1), targets.flatten())
        value = float(loss.detach())
        forward_seconds = time.monotonic() - started
        if not math.isfinite(value):
            raise RuntimeError('non-finite straight-through planning goal CE')
        history.append(dict(iteration=iteration, loss=value, goal_ce=value))
        diag.event('planning_forward_end', round=round_index, iteration=iteration,
                   iterations=iterations, horizon=horizon, goal_ce=value,
                   forward_seconds=forward_seconds, mode='straight_through')
        if iteration % 10 == 0 or iteration == iterations:
            print0(f'Planning progress: round={round_index + 1} iteration={iteration}/{iterations} '
                   f'horizon={horizon} {metric}={value:.6f} mode=straight_through', flush=True)
        if value < best_loss:
            best_loss, best_actions = value, actions.detach().clone()
        if iteration < iterations:
            optimizer.zero_grad(set_to_none=True)
            backward_started = time.monotonic()
            loss.backward()
            backward_seconds = time.monotonic() - backward_started
            update_started = time.monotonic()
            torch.nn.utils.clip_grad_norm_([actions], 1.0)
            optimizer.step()
            _dispatch(actions.device, wait=True)
            optimizer_seconds = time.monotonic() - update_started
        else:
            backward_seconds = optimizer_seconds = 0.0
        diag.event('planning_iteration_end', round=round_index, iteration=iteration,
                   iterations=iterations, horizon=horizon, goal_ce=value, mode='straight_through',
                   elapsed=time.monotonic() - started, forward_seconds=forward_seconds,
                   backward_seconds=backward_seconds, optimizer_seconds=optimizer_seconds)
        if iteration < 3 or iteration % 10 == 0 or iteration == iterations:
            print0(f'Planning timing: round={round_index + 1} iteration={iteration}/{iterations} '
                   f'forward={forward_seconds:.2f}s backward={backward_seconds:.2f}s '
                   f'update={optimizer_seconds:.2f}s', flush=True)
        if actions.device.type == 'xla' and (iteration < 3 or iteration % 10 == 0):
            diag.event('planning_xla_metrics', round=round_index, iteration=iteration,
                       metrics=diag.xla_metrics(actions.device))
    return dict(actions=best_actions, loss=best_loss, initial_loss=history[0]['loss'], history=history)


def st_plan(model, prompt, goal, horizon=20, execute_steps=None, iterations=100,
            lr=0.01, seed=123, return_logits=False):
    execute_steps = horizon if execute_steps is None else execute_steps
    if (prompt.ndim != 2 or prompt.size(1) < 1 or goal.ndim != 1
            or not 0 < goal.numel() <= horizon or not 1 <= execute_steps <= horizon
            or iterations < 1 or not math.isfinite(lr) or lr <= 0):
        raise ValueError('invalid prompt, goal, horizon, iterations or planning learning rate')
    if horizon >= model.config.sequence_len:
        raise ValueError('planning horizon must leave at least one prompt position')
    training, flags = _freeze_model(model)
    try:
        prompt = _trim_prompt(prompt, model.config.sequence_len - horizon)
        cache = DifferentiableKVCache(model.config.n_layer, prompt.size(1) + horizon)
        with torch.no_grad():
            initial = model.encode(prompt, kv_cache=cache)[:, -1].detach()
        generated, executed_logits, rounds, history = [], [], [], []
        warm, previous_count = None, 0
        initial_loss = None
        for start in range(0, horizon, execute_steps):
            remaining, count = horizon - start, min(execute_steps, horizon - start)
            round_goal = goal if remaining >= goal.numel() else goal[-remaining:]
            action_init = None if warm is None else warm[:, previous_count:].detach().clone()
            optimized = optimize_st_actions(model, initial, cache, round_goal, remaining,
                                             iterations, lr, seed + start,
                                             action_init=action_init, round_index=len(rounds))
            initial_loss = optimized['initial_loss'] if initial_loss is None else initial_loss
            history.extend([dict(round=len(rounds), **r) for r in optimized['history']])
            diag.event('planning_execute_begin', round=len(rounds), start=start, count=count)
            with torch.no_grad():
                for action in optimized['actions'][:, :count].unbind(1):
                    logits = model.predict_logits(initial, action)
                    token = logits.argmax(-1)
                    generated.append(token)
                    executed_logits.append(logits)
                    initial = model.encode(token[:, None], kv_cache=cache)[:, -1].detach()
                    # Committing ten tokens must not fuse ten complete models
                    # into one lazy graph before the next host read.
                    _dispatch(initial.device)
            diag.event('planning_execute_end', round=len(rounds), start=start, count=count)
            rounds.append(dict(start=start, horizon=remaining, executed=count,
                               initial_loss=optimized['initial_loss'], goal_ce=optimized['loss'], goal_mse=optimized['loss']))
            warm, previous_count = optimized['actions'], count
        tokens = torch.stack(generated, dim=1)
        goal_tokens = tokens[:, -goal.numel():]
        exact = goal_tokens.eq(goal[None]).all(-1).float().mean()
        result = dict(tokens=tokens.cpu(), goal_tokens=goal_tokens.cpu(),
                      goal_exact_match=float(exact), initial_loss=initial_loss,
                      goal_ce=optimized['loss'], goal_loss=optimized['loss'],
                      best_loss=optimized['loss'], goal_objective='goal_mse' if model.config.token_objective == 'embedding_mse' else 'goal_ce',
                      goal_mse=optimized['loss'],
                      planning_mode='straight_through', round_history=rounds, history=history)
        if return_logits:
            result['logits'] = torch.stack(executed_logits, dim=1).detach().cpu()
        return result
    finally:
        _restore_model(model, training, flags)
