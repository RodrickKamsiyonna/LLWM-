"""Portable attention checks and opt-in TPU integration coverage."""
import os
import pytest
import torch
from nanochat.flash_attention import _xla_attention


@pytest.mark.parametrize("length,kv_length,window,heads", [(5,5,-1,2), (5,5,2,4), (1,7,3,4), (3,7,-1,2)])
def test_xla_attention_matches_sdpa(length, kv_length, window, heads):
    torch.manual_seed(42)
    q = torch.randn(2, heads, length, 8)
    k = torch.randn(2, 2, kv_length, 8)
    v = torch.randn_like(k)
    actual = _xla_attention(q, k, v, (window, -1))
    rows = torch.arange(length)[:, None] + kv_length - length
    cols = torch.arange(kv_length)[None, :]
    mask = cols <= rows
    if window >= 0:
        mask &= rows - cols <= window
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(heads // 2, dim=1),
        v.repeat_interleave(heads // 2, dim=1), attn_mask=mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(os.environ.get("PJRT_DEVICE") != "TPU", reason="requires TPU")
@pytest.mark.parametrize("architecture", ["gpt", "llwm"])
def test_tpu_training_checkpoint_and_generation(tmp_path, architecture):
    from nanochat.common import compute_init
    from nanochat.gpt import GPTConfig
    from nanochat.llwm import LLWMConfig
    from nanochat.model import create_model
    from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
    _, rank, _, world, device = compute_init("xla")
    assert world == 1, "Run this pytest test on a single device"
    config_cls = LLWMConfig if architecture == "llwm" else GPTConfig
    model = create_model(config_cls(sequence_len=16, vocab_size=128, n_layer=2,
                                   n_head=2, n_kv_head=1, n_embd=64, window_pattern="L"))
    model = model.to(device)
    model.init_weights()
    optimizer = model.setup_optimizer()
    ids = torch.randint(0, 128, (2, 8), device=device)
    for _ in range(2):
        loss = model(ids, ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert torch.isfinite(loss).item()
    save_checkpoint(str(tmp_path), 2, model.state_dict(), optimizer.state_dict(), {}, rank)
    weights, state, _ = load_checkpoint(str(tmp_path), 2, device, True, rank)
    restored = create_model(model.config).to(device)
    restored.init_weights()
    restored.load_state_dict(weights)
    restored_optimizer = restored.setup_optimizer()
    restored_optimizer.load_state_dict(state)
    restored(ids, ids).backward()
    restored_optimizer.step()
    restored_optimizer.zero_grad(set_to_none=True)
    assert len(list(restored.generate([1, 2], max_tokens=2, seed=42))) == 2
    # Cached decoding must match full-context logits.
    from nanochat.engine import KVCache
    from nanochat.common import COMPUTE_DTYPE
    cache = KVCache(batch_size=1, seq_len=16, num_heads=1,
                    head_dim=32, num_layers=2, device=device, dtype=COMPUTE_DTYPE)
    with torch.inference_mode():
        restored(ids[:1, :3], kv_cache=cache)
        cached = restored(ids[:1, 3:4], kv_cache=cache)[:, -1]
        full = restored(ids[:1, :4])[:, -1]
    torch.testing.assert_close(cached.cpu(), full.cpu(), atol=0.05, rtol=0.05)


def test_uncompiled_adamw_matches_torch():
    """The XLA kernel path must preserve AdamW math without Inductor."""
    from nanochat.optim import adamw_step_fused
    torch.manual_seed(7)
    actual = torch.randn(4, 8)
    reference = torch.nn.Parameter(actual.clone())
    optimizer = torch.optim.AdamW([reference], lr=0.01, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.1)
    moment, variance = torch.zeros_like(actual), torch.zeros_like(actual)
    scalar = lambda x: torch.tensor(float(x))
    for step in range(1, 4):
        gradient = torch.randn_like(actual)
        reference.grad = gradient.clone()
        optimizer.step()
        adamw_step_fused.__wrapped__(actual, gradient, moment, variance,
                                    scalar(step), scalar(0.01), scalar(0.8),
                                    scalar(0.95), scalar(1e-8), scalar(0.1))
        torch.testing.assert_close(actual, reference.detach(), atol=1e-6, rtol=1e-5)
