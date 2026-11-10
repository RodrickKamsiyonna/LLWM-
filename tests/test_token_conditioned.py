"""Token posterior alignment and straight-through, cached planning gradients."""
from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from nanochat.engine import DifferentiableKVCache
from nanochat.llwm import LLWM, LLWMConfig
from nanochat.model import config_from_dict, create_model
from nanochat.planning import mpc_plan
from nanochat.st_planning import st_rollout


def make_token_model():
    torch.manual_seed(19)
    model = LLWM(LLWMConfig(sequence_len=16, vocab_size=16, n_layer=2,
                            n_embd=32, n_head=2, n_kv_head=1, window_pattern='L',
                            action_dim=8, action_layers=2, predictor_layers=6,
                            prediction_horizon=1, head_source='predictor', head_training='joint',
                            action_conditioning='next_token', planning_mode='straight_through',
                            memory_weight=0., kl_mode='target', target_kl=1., kl_lambda=3.))
    model.init_weights()
    return model


def test_token_posterior_and_next_token_loss_without_future_encoder_pass():
    model = make_token_model()
    idx, target = torch.tensor([[1,2,3]]), torch.tensor([[2,3,4]])
    posterior_inputs, compressor_calls, predictions = [], [], []
    hooks = [model.action_encoder.register_forward_pre_hook(lambda m,a: posterior_inputs.append(a[0])),
             model.compressor_norm.register_forward_hook(lambda m,a,o: compressor_calls.append(o)),
             model.predictor_norm.register_forward_hook(lambda m,a,o: predictions.append(o))]
    v = model(idx, target, return_loss_components=True)
    for hook in hooks:
        hook.remove()
    assert len(compressor_calls) == len(predictions) == len(posterior_inputs) == 1
    torch.testing.assert_close(posterior_inputs[0][..., :32], compressor_calls[0])
    torch.testing.assert_close(posterior_inputs[0][..., 32:], model.transformer.wte(target))
    expected = F.cross_entropy(model.decode_memory(predictions[0]).flatten(0,1), target.flatten())
    torch.testing.assert_close(v['ce'], expected)
    assert v['memory_loss'].item() == 0
    torch.testing.assert_close(v['loss'], v['ce'] + 3 * F.relu(v['kl_total'] - 1).square())
    v['loss'].backward()
    for module in (model.compressor, model.action_encoder, model.predictor, model.llm_head_mlp, model.llm_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_functional_cache_matches_full_compressor_and_fork_is_immutable():
    model = make_token_model().eval()
    tokens = torch.tensor([[1,2,3,4,5]])
    cache = DifferentiableKVCache(model.config.n_layer, 16)
    with torch.no_grad():
        expected = model.encode(tokens)
        prefix = model.encode(tokens[:, :2], kv_cache=cache)
        fork = cache.fork()
        pieces = [prefix] + [model.encode(tokens[:, i:i+1], kv_cache=fork) for i in range(2,5)]
    torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=1e-6, rtol=1e-5)
    assert cache.get_pos() == 2 and fork.get_pos() == 5
    assert all(k.size(1) == 2 for k in cache.keys)


def test_st_token_embedding_forward_is_hard_and_backward_is_nonzero():
    model = make_token_model().eval()
    cache = DifferentiableKVCache(2, 16)
    with torch.no_grad():
        model.encode(torch.tensor([[1,2]]), kv_cache=cache)
    raw = torch.randn(1,1,16,requires_grad=True)
    probs = raw.softmax(-1)
    ids = probs.argmax(-1)
    st = model.encode_st_tokens(probs, ids, kv_cache=cache.fork())
    with torch.no_grad():
        hard = model.encode(ids, kv_cache=cache.fork())
    torch.testing.assert_close(st, hard, atol=1e-6, rtol=1e-5)
    grad = torch.autograd.grad(st[..., 0].sum(), raw)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_st_rollout_matches_hard_tokens_and_rematerialized_gradients():
    model = make_token_model().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    prompt = torch.tensor([[1,2,3]])
    cache = DifferentiableKVCache(2, 16)
    with torch.no_grad():
        initial = model.encode(prompt, kv_cache=cache)[:, -1]
    actions = torch.randn(1,4,8,requires_grad=True)
    logits = st_rollout(model, initial, cache, actions, checkpoint_steps=False)
    with torch.no_grad():
        context, hard_logits = prompt, []
        for action in actions.unbind(1):
            memory = model.encode(context)[:, -1]
            out = model.predict_logits(memory, action)
            hard_logits.append(out)
            context = torch.cat((context, out.argmax(-1,keepdim=True)), 1)
    torch.testing.assert_close(logits, torch.stack(hard_logits,1), atol=1e-6, rtol=1e-5)
    loss = F.cross_entropy(logits[:,-1], torch.tensor([4]))
    grad = torch.autograd.grad(loss, actions)[0]
    assert torch.isfinite(grad).all() and grad[:,0].abs().sum() > 0
    replay = actions.detach().clone().requires_grad_()
    replay_logits = st_rollout(model, initial, cache, replay, checkpoint_steps=True)
    replay_grad = torch.autograd.grad(F.cross_entropy(replay_logits[:,-1],torch.tensor([4])),replay)[0]
    torch.testing.assert_close(replay_logits, logits)
    torch.testing.assert_close(replay_grad, grad, atol=1e-6, rtol=1e-5)
    assert cache.get_pos() == 3 and all(k.size(1)==3 for k in cache.keys)


def test_st_mpc_freezes_weights_and_reuses_warm_start():
    model = make_token_model()
    weights = {n:p.detach().clone() for n,p in model.named_parameters()}
    result = mpc_plan(model, torch.tensor([[1,2,3]]), torch.tensor([4]),
                      horizon=4, execute_steps=2, iterations=3, lr=.01, return_logits=True)
    assert result['planning_mode'] == 'straight_through'
    assert result['tokens'].shape == (1,4) and result['logits'].shape == (1,4,16)
    assert [r['horizon'] for r in result['round_history']] == [4,2]
    assert [r['executed'] for r in result['round_history']] == [2,2]
    assert all(r['goal_ce'] <= r['initial_loss'] for r in result['round_history'])
    for n,p in model.named_parameters():
        assert p.requires_grad and p.grad is None
        torch.testing.assert_close(p, weights[n])
    assert model.training


@pytest.mark.parametrize('goal_count', [1, 2])
def test_stagewise_vjp_matches_full_st_gradient_including_cache_and_smear(goal_count):
    model = make_token_model().eval()
    with torch.no_grad():
        model.smear_lambda.fill_(.3)
    for p in model.parameters():
        p.requires_grad_(False)
    cache = DifferentiableKVCache(2,16)
    with torch.no_grad():
        initial = model.encode(torch.tensor([[1,2,3]]),kv_cache=cache)[:,-1]
    actions = torch.randn(1,4,8,requires_grad=True)
    reference = st_rollout(model,initial,cache,actions,checkpoint_steps=False,stagewise=False)
    targets = torch.tensor([[4,5]])[:,-goal_count:]
    expected_grad = torch.autograd.grad(
        F.cross_entropy(reference[:,-goal_count:].flatten(0,1),targets.flatten()),actions)[0]
    staged_actions = actions.detach().clone().requires_grad_()
    staged = st_rollout(model,initial,cache,staged_actions,stagewise=True)
    actual_grad = torch.autograd.grad(
        F.cross_entropy(staged[:,-goal_count:].flatten(0,1),targets.flatten()),staged_actions)[0]
    torch.testing.assert_close(staged,reference,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(actual_grad,expected_grad,atol=1e-6,rtol=1e-5)
    assert actual_grad[:,0].abs().sum()>0
    assert cache.get_pos()==3 and all(k.size(1)==3 for k in cache.keys)


def test_token_architecture_meta_roundtrip_and_masked_loss():
    model = make_token_model()
    with torch.device('meta'):
        copy = create_model(config_from_dict(asdict(model.config)))
    copy.to_empty(device='cpu'); copy.init_weights(); copy.load_state_dict(model.state_dict())
    tokens = torch.tensor([[1,2,3]])
    torch.testing.assert_close(model(tokens), copy(tokens))
    ignored = model(tokens, torch.full_like(tokens,-1), return_loss_components=True)
    assert all(v.item() == 0 for v in ignored.values())


@pytest.mark.parametrize('bad', [dict(head_source='compressor'), dict(memory_weight=1), dict(prediction_horizon=8)])
def test_token_architecture_rejects_conflicting_legacy_settings(bad):
    cfg = asdict(make_token_model().config)
    cfg.update(bad)
    with pytest.raises(ValueError):
        LLWMConfig(**cfg)
