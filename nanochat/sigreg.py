"""FP32 token-level SIGReg, following LeJEPA's published minimal implementation.

https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
Statistics use valid tokens within each rank/microbatch; gradients are averaged
by the existing replicated optimizer. No extra distributed collectives.
"""
import torch
import torch.nn.functional as F


def sigreg_loss(memory, valid, projections=1024, knots=17):
    x = memory.float().flatten(0, 1)
    mask = valid.flatten().float()
    count = mask.sum().clamp(min=1)
    directions = F.normalize(torch.randn(x.size(-1), projections, device=x.device,
                                          dtype=torch.float32), dim=0)
    t = torch.linspace(0, 3, knots, device=x.device, dtype=torch.float32)
    phi = torch.exp(-t.square() / 2)
    weights = torch.ones_like(t) * (6 / (knots - 1))
    weights = weights * torch.where((t == 0) | (t == 3), 0.5, 1.) * phi
    # Bound trigonometric workspace by processing sketches in groups of 32.
    projected = x @ directions
    total = x.new_zeros(())
    for block in projected.split(32, dim=-1):
        phase = block.unsqueeze(-1) * t
        real = (phase.cos() * mask[:, None, None]).sum(0) / count
        imag = (phase.sin() * mask[:, None, None]).sum(0) / count
        total = total + (((real - phi).square() + imag.square()) @ weights).sum()
    return total * count / projections
