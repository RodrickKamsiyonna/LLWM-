"""Exercise LLWM training and RL on synthetic data, on CPU/CUDA or all TPU replicas.

python -m scripts.xla_launch scripts.llwm_smoke
"""
import torch
from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, print0
from nanochat.llwm import LLWM, LLWMConfig


def main():
    _, rank, _, world, device = compute_init(autodetect_device_type())
    config = LLWMConfig(sequence_len=16, vocab_size=128, n_layer=4, n_embd=64,
                        n_head=2, n_kv_head=1, action_dim=16, window_pattern='L')
    model = LLWM(config).to(device)
    model.init_weights()
    if device.type == 'xla':
        import torch_xla.core.xla_model as xm
        xm.broadcast_master_param(model)
    optimizer = model.setup_optimizer()
    generator = torch.Generator().manual_seed(42 + rank)
    tokens = torch.randint(128, (2,9), generator=generator).to(device)
    idx, targets = tokens[:,:-1], tokens[:,1:]
    for _ in range(2):
        values = model(idx, targets, return_loss_components=True)
        values['loss'].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert all(torch.isfinite(value).item() for value in values.values())
    # RL's causal policy leaves posterior parameters untouched.
    before = [p.detach().clone() for p in model.action_encoder.parameters()]
    model(idx, targets, loss_reduction='none').mean().backward()
    optimizer.step()
    for param, expected in zip(model.action_encoder.parameters(), before):
        torch.testing.assert_close(param.cpu(), expected.cpu())
    if device.type == 'xla':
        import torch_xla
        flat = torch.cat([p.detach().float().flatten() for p in model.parameters()])
        replicas = xm.all_gather(flat.unsqueeze(0), dim=0)
        torch_xla.sync(wait=True)
        replicas = replicas.cpu()
        torch.testing.assert_close(replicas, replicas[:1].expand_as(replicas))
    print0(f'LLWM smoke passed: {config.n_layer} compressor layers, {config.predictor_layers} predictor layers, {world} replicas')
    compute_cleanup()


if __name__ == '__main__':
    main()
