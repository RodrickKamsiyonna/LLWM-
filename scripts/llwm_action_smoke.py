"""Synthetic action-head training and goal-memory MPC check on all TPU replicas."""
import torch

from nanochat.common import compute_init, compute_cleanup, print0
from nanochat import diagnostics as diag
from nanochat.distributed_utils import require_equal
from nanochat.llwm import LLWM, LLWMConfig
from nanochat.planning import mpc_plan


def main():
    import torch_xla
    import torch_xla.core.xla_model as xm

    _, rank, _, world, device = compute_init('xla')
    config = LLWMConfig(sequence_len=32, vocab_size=128, n_layer=2, n_embd=64,
                        n_head=2, n_kv_head=1, action_dim=128, action_layers=2,
                        predictor_layers=3, prediction_horizon=4, window_pattern='L',
                        head_source='action', head_training='joint',
                        target_kl=2.0, kl_lambda=3.0, kl_mode='target')
    model = LLWM(config).to(device)
    model.init_weights()
    xm.broadcast_master_param(model)
    optimizer = model.setup_optimizer()
    generator = torch.Generator().manual_seed(42 + rank)
    tokens = torch.randint(128, (1, 5), generator=generator).to(device)
    for step in range(2):
        values = model(tokens[:, :-1], tokens[:, 1:], return_loss_components=True)
        values['loss'].backward()
        torch_xla.sync(wait=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch_xla.sync(wait=True)
        loss = float(values['loss'].detach())
        assert torch.isfinite(torch.tensor(loss))
        diag.event('action_head_smoke_step_complete', step_number=step + 1, loss=loss)

    flat = torch.cat([p.detach().float().flatten() for p in model.parameters()])
    replicas = xm.all_gather(flat.unsqueeze(0), dim=0)
    torch_xla.sync(wait=True)
    replicas = replicas.cpu()
    torch.testing.assert_close(replicas, replicas[:1].expand_as(replicas), atol=1e-5, rtol=1e-5)
    del flat, replicas, values
    require_equal('trained', 'action-head-smoke-before-plan')
    if rank == 0:
        result = mpc_plan(model, torch.tensor([[1, 2, 3]], device=device),
                          torch.tensor([4, 5], device=device), horizon=20,
                          execute_steps=10, iterations=2, lr=0.01, return_logits=True)
        assert result['goal_objective'] == 'goal_memory_loss'
        assert result['tokens'].shape == (1, 20)
        assert [r['executed'] for r in result['round_history']] == [10, 10]
        assert all(r['goal_memory_loss'] <= r['initial_loss'] for r in result['round_history'])
        print0(f'Action-head goal-memory MPC passed: {result["round_history"]}', flush=True)
    require_equal('planned', 'action-head-smoke-after-plan')
    diag.event('action_head_smoke_passed', world_size=world)
    print0(f'Action-head optimizer and replica checks passed on {world} TPU devices', flush=True)
    compute_cleanup()


if __name__ == '__main__':
    main()
