"""Synthetic multi-replica optimizer check; no tokenizer or dataset required.

python -m scripts.xla_launch scripts.xla_smoke
"""
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from nanochat.common import compute_init, compute_cleanup, print0
from nanochat.optim import MuonAdamW


def main():
    _, rank, _, world, device = compute_init("xla")
    adam = torch.nn.Parameter(torch.ones(8, device=device))
    matrix = torch.nn.Parameter(torch.ones(8, 8, device=device))
    optimizer = MuonAdamW([
        dict(kind="adamw", params=[adam], lr=0.01, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.0),
        dict(kind="muon", params=[matrix], lr=0.01, momentum=0.95, beta2=0.9, ns_steps=5, weight_decay=0.0),
    ])
    # Different rank gradients must produce the same averaged update everywhere.
    for _ in range(2):
        adam.grad = torch.full_like(adam, float(rank + 1))
        matrix.grad = torch.full_like(matrix, float(rank + 1))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.testing.assert_close(adam.cpu(), torch.full((8,), 0.98), atol=1e-5, rtol=1e-5)
    for param in (adam, matrix):
        replicas = xm.all_gather(param.detach().reshape(1, -1), dim=0)
        torch_xla.sync(wait=True)
        replicas = replicas.cpu()
        assert torch.isfinite(replicas).all()
        torch.testing.assert_close(replicas, replicas[:1].expand_as(replicas))
    # The first moment verifies gradient averaging, not just parameter agreement.
    expected_moment = ((world + 1) / 2) * (1 - 0.8 ** 2)
    torch.testing.assert_close(optimizer.state[adam]["exp_avg"].cpu(),
                               torch.full((8,), expected_moment), atol=1e-5, rtol=1e-5)
    print0(f"TPU optimizer smoke passed on {world} replicas")
    compute_cleanup()


if __name__ == "__main__":
    main()
