"""Predictor-only latent-action planning and receding-horizon MPC."""
import math

import torch
from torch.nn import functional as F


def rollout_memories(model, memory, actions):
    """Unroll the learned predictor; imagined steps never invoke the compressor."""
    if memory.ndim == 3:
        memory = memory[:, -1]
    if memory.ndim != 2 or actions.ndim != 3 or memory.size(0) != actions.size(0):
        raise ValueError('memory must be [batch,width]; actions [batch,horizon,action_dim]')
    states = []
    checkpoint_predictor = (memory.device.type == 'xla' and torch.is_grad_enabled()
                            and actions.requires_grad)
    if checkpoint_predictor:
        # Planning differentiates through the predictor while model weights
        # are frozen. Rematerialize each recurrent transition on backward so
        # the 20-step rollout does not retain every layer's activations beside
        # the sharded training optimizer state on the TPU.
        # XLA provides its own checkpoint implementation. PyTorch's generic
        # non-reentrant path expects torch.xla, which torch/XLA does not expose.
        from torch_xla.utils.checkpoint import checkpoint
    for index, action in enumerate(actions.unbind(1)):
        if checkpoint_predictor:
            memory = checkpoint(model.predict_features, memory, action, use_reentrant=True)
        else:
            memory = model.predict_features(memory, action)
        states.append(memory)
        # Keep XLA programs bounded: one 20-step rollout would otherwise trace
        # 20 copies of the full predictor into a single lazy graph. Dispatch
        # each 4-step segment; this preserves the recurrent autograd path.
        if memory.device.type == 'xla' and ((index + 1) % 4 == 0 or index + 1 == actions.size(1)):
            import torch_xla
            torch_xla.sync(wait=False)
    return torch.stack(states, dim=1)


def rollout(model, memory, actions):
    """Decode actions directly for action heads; predicted memories otherwise."""
    if model.config.head_source == 'action':
        return model.decode_actions(actions)
    return model.decode_memory(rollout_memories(model, memory, actions))


def encode_goal_memory(model, goal, batch_size):
    """Standalone goal context; never insert goal tokens into the prompt cache."""
    with torch.no_grad():
        return model.encode(goal[None].expand(batch_size, -1))[:, -1].detach()


def _optimize_actions(model, initial, goal, horizon, iterations, lr, seed, action_init=None,
                      round_index=0, goal_memory=None):
    from nanochat import diagnostics as diag
    from nanochat.common import print0
    targets = goal[None].expand(initial.size(0), -1)
    memory_objective = model.config.head_source == 'action'
    metric_name = 'goal_memory_loss' if memory_objective else 'goal_ce'
    if memory_objective:
        goal_memory = (encode_goal_memory(model, goal, initial.size(0))
                       if goal_memory is None else goal_memory.detach())
        if goal_memory.shape != initial.shape:
            raise ValueError('goal memory must match the initial [batch,width] state')
    if action_init is None:
        action_init = sample_actions(initial.size(0), horizon, model.config.action_dim,
                                     device=initial.device, seed=seed)
    if action_init.shape != (initial.size(0), horizon, model.config.action_dim):
        raise ValueError('warm-start actions have the wrong shape')
    actions = torch.nn.Parameter(action_init.detach().clone())
    # Keep Adam's timestep/bias correction on XLA. A Python timestep bakes a
    # new bias-correction constant into the lazy graph on every iteration.
    optimizer = torch.optim.Adam([actions], lr=lr, capturable=actions.device.type == 'xla')
    history, best_loss, best_actions = [], float('inf'), None
    for iteration in range(iterations + 1):
        diag.event('planning_iteration_begin', round=round_index, iteration=iteration,
                   iterations=iterations, horizon=horizon)
        with torch.set_grad_enabled(iteration < iterations):
            states = rollout_memories(model, initial, actions)
            if memory_objective:
                loss = F.smooth_l1_loss(states[:, -1].float(), goal_memory.float())
            else:
                logits = model.decode_memory(states[:, -goal.numel():])
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        value = float(loss.detach())
        diag.event('planning_iteration_end', round=round_index, iteration=iteration,
                   iterations=iterations, horizon=horizon, **{metric_name: value})
        if iteration % 10 == 0 or iteration == iterations:
            print0(f'Planning progress: round={round_index + 1} iteration={iteration}/{iterations} '
                   f'horizon={horizon} {metric_name}={value:.6f}', flush=True)
        if not math.isfinite(value):
            raise RuntimeError('non-finite planning objective')
        history.append({'iteration': iteration, 'loss': value, metric_name: value})
        if value < best_loss:
            best_loss, best_actions = value, actions.detach().clone()
        if iteration < iterations:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([actions], 1.0)
            optimizer.step()
            if actions.device.type == 'xla':
                import torch_xla
                torch_xla.sync(wait=False)
    return {'actions': best_actions, 'loss': best_loss,
            'initial_loss': history[0]['loss'], 'history': history,
            'metric_name': metric_name}


def _freeze_model(model):
    training = model.training
    flags = [parameter.requires_grad for parameter in model.parameters()]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return training, flags


def _restore_model(model, training, flags):
    for parameter, flag in zip(model.parameters(), flags):
        parameter.requires_grad_(flag)
    model.train(training)


def _trim_prompt(prompt, context):
    """Keep the most recent context tokens without an out-of-range negative slice."""
    start = max(0, prompt.size(1) - context)
    return prompt[:, start:].detach()


def plan(model, prompt, goal, horizon=500, iterations=100, lr=0.05, seed=123,
         return_logits=False):
    """Optimize terminal goal-memory distance for action heads, goal CE otherwise."""
    if model.config.planning_mode == 'straight_through':
        from nanochat.st_planning import st_plan
        return st_plan(model, prompt, goal, horizon=horizon, iterations=iterations,
                       lr=lr, seed=seed, return_logits=return_logits)
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('planning lr must be finite and positive')
    if goal.ndim != 1 or goal.numel() < 1 or horizon < goal.numel() or iterations < 1:
        raise ValueError('goal must be nonempty; horizon >= goal length; iterations >= 1')
    if prompt.ndim != 2 or prompt.size(1) < 1:
        raise ValueError('prompt must be [batch, nonempty sequence]')
    if horizon >= model.config.sequence_len:
        raise ValueError('planning horizon must leave one context position for the prompt')
    training, flags = _freeze_model(model)
    try:
        with torch.no_grad():
            context = model.config.sequence_len - horizon
            memory = model.encode(_trim_prompt(prompt, context))[:, -1].detach()
        optimized = _optimize_actions(model, memory, goal, horizon, iterations, lr, seed)
        with torch.no_grad():
            logits = rollout(model, memory, optimized['actions'])
            tokens = logits.argmax(-1)
            goal_tokens = tokens[:, -goal.numel():]
            targets = goal[None].expand(memory.size(0), -1)
            exact = goal_tokens.eq(targets).all(-1).float().mean()
        result = {'tokens': tokens.cpu(), 'goal_tokens': goal_tokens.cpu(),
                  'goal_exact_match': exact.item(), 'history': optimized['history'],
                  'best_loss': optimized['loss'], 'goal_loss': optimized['loss'],
                  'goal_objective': optimized['metric_name'],
                  optimized['metric_name']: optimized['loss']}
        if return_logits:
            result['logits'] = logits.detach().cpu()
        return result
    finally:
        _restore_model(model, training, flags)


def mpc_plan(model, prompt, goal, horizon=50, execute_steps=5, iterations=100,
             lr=0.05, seed=123, return_logits=False):
    """Replan with predictor-only imagined rollouts and execute token blocks.

    The compressor encodes the real prompt once and then only the tokens that
    MPC actually commits. Action-head planning also encodes the goal once in a
    separate context and minimizes terminal-memory SmoothL1. Imagined future
    states always use the predictor alone.
    """
    if model.config.planning_mode == 'straight_through':
        from nanochat.st_planning import st_plan
        return st_plan(model, prompt, goal, horizon=horizon, execute_steps=execute_steps,
                       iterations=iterations, lr=lr, seed=seed, return_logits=return_logits)
    if not 1 <= execute_steps <= horizon or not 0 < goal.numel() <= horizon:
        raise ValueError('invalid execution horizon or goal length')
    if prompt.ndim != 2 or prompt.size(1) < 1 or goal.ndim != 1 or iterations < 1:
        raise ValueError('invalid prompt, goal or iterations')
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('planning lr must be finite and positive')
    if horizon >= model.config.sequence_len:
        raise ValueError('planning horizon must leave one context position for the prompt')

    training, flags = _freeze_model(model)
    try:
        from nanochat.engine import KVCache

        context = model.config.sequence_len - horizon
        prompt = _trim_prompt(prompt, context)
        cache = KVCache(batch_size=prompt.size(0), num_heads=model.config.n_kv_head,
                        seq_len=prompt.size(1) + horizon,
                        head_dim=model.config.n_embd // model.config.n_head,
                        num_layers=model.config.n_layer, device=prompt.device,
                        dtype=model.transformer.wte.weight.dtype)
        with torch.no_grad():
            initial = model.encode(prompt, kv_cache=cache)[:, -1].detach()
            goal_memory = (encode_goal_memory(model, goal, prompt.size(0))
                           if model.config.head_source == 'action' else None)

        generated, executed_logits, rounds, history = [], [], [], []
        warm, previous_count = None, 0
        initial_loss = None
        best_loss = float('inf')
        for start in range(0, horizon, execute_steps):
            remaining = horizon - start
            count = min(execute_steps, remaining)
            round_goal = goal if remaining >= goal.numel() else goal[-remaining:]
            action_init = None
            if warm is None:
                round_seed = seed + start
            else:
                action_init = warm[:, previous_count:].detach().clone()
                round_seed = seed + start
            optimized = _optimize_actions(model, initial, round_goal, remaining, iterations, lr,
                                          round_seed, action_init=action_init,
                                          round_index=len(rounds), goal_memory=goal_memory)
            actions = optimized['actions']
            initial_loss = optimized['initial_loss'] if initial_loss is None else initial_loss
            best_loss = optimized['loss']
            history.extend([{'round': len(rounds), **item} for item in optimized['history']])
            with torch.no_grad():
                for index in range(count):
                    action = actions[:, index]
                    logits = model.predict_logits(initial, action)
                    token = logits.argmax(-1)
                    executed_logits.append(logits)
                    generated.append(token)
                    # Update only the real state, using the inference KV cache.
                    initial = model.encode(token[:, None], kv_cache=cache)[:, -1].detach()
            rounds.append({'start': start, 'horizon': remaining, 'executed': count,
                           'initial_loss': optimized['initial_loss'],
                           optimized['metric_name']: optimized['loss']})
            warm, previous_count = actions, count

        tokens = torch.stack(generated, dim=1)
        goal_tokens = tokens[:, -goal.numel():]
        targets = goal[None].expand(tokens.size(0), -1)
        exact = goal_tokens.eq(targets).all(-1).float().mean()
        result = {'tokens': tokens.cpu(), 'goal_tokens': goal_tokens.cpu(),
                  'goal_exact_match': exact.item(), 'initial_loss': initial_loss,
                  'goal_loss': best_loss, 'goal_objective': optimized['metric_name'],
                  optimized['metric_name']: best_loss, 'best_loss': best_loss,
                  'round_history': rounds, 'history': history}
        if return_logits:
            result['logits'] = torch.stack(executed_logits, dim=1).detach().cpu()
        return result
    finally:
        _restore_model(model, training, flags)


def sample_actions(batch_size, horizon, action_dim, *, device, seed=123):
    """Independently initialize all candidate actions from N(0, I)."""
    device = torch.device(device)
    rng_device = 'cpu' if device.type == 'xla' else device
    generator = torch.Generator(device=rng_device).manual_seed(seed)
    return torch.randn(batch_size, horizon, action_dim, generator=generator,
                       device=rng_device).to(device)
