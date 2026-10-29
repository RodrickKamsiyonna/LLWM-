"""Small startup/checkpoint control collectives; never reduce training gradients here."""
import json
import os
import torch
import torch.distributed as dist


def gather_json(value, tag):
    if os.environ.get('PJRT_DEVICE') == 'TPU':
        import torch_xla.runtime as xr
        if xr.world_size() > 1:
            import torch_xla.core.xla_model as xm
            # JSON avoids torch.load/pickle object rendezvous and weights_only changes.
            payloads = xm.rendezvous(tag, json.dumps(value, sort_keys=True).encode())
            return [json.loads(payload) for payload in payloads]
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        values = [None] * dist.get_world_size()
        dist.all_gather_object(values, value)
        return values
    return [value]


def require_equal(value, tag):
    values = gather_json(value, tag)
    canonical = [json.dumps(v, sort_keys=True) for v in values]
    if len(set(canonical)) != 1:
        raise RuntimeError(f'Distributed mismatch for {tag}: {values}')
    return values


def accumulation_steps(total_tokens, batch_size, sequence_len, world_size):
    values = (total_tokens, batch_size, sequence_len, world_size)
    if any(v <= 0 for v in values):
        raise ValueError(f'Token batch, batch size, context and world size must be positive: {values}')
    per_microstep = batch_size * sequence_len * world_size
    if total_tokens % per_microstep:
        raise ValueError(f'total_batch_size={total_tokens} must be divisible by {per_microstep} (B*T*world)')
    return total_tokens // per_microstep
