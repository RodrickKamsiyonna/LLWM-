"""Numerical checks for the LLWM architecture and training/inference contract."""
from dataclasses import asdict
import io
import pytest
import torch
from torch.nn import functional as F
from nanochat.llwm import ActionEncoder, LLWM, LLWMConfig
from nanochat.model import config_from_dict, create_model


@pytest.fixture(autouse=True)
def portable_torch(monkeypatch):
    # Intel macOS only has torch 2.2 wheels. Equivalent primitives allow local
    # numerical testing; production uses torch/XLA 2.9 without these shims.
    torch.set_num_threads(1)
    if not hasattr(F, 'rms_norm'):
        def rms_norm(x, shape, weight=None, eps=None):
            eps = torch.finfo(x.dtype).eps if eps is None else eps
            y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
            return y if weight is None else y * weight
        monkeypatch.setattr(F, 'rms_norm', rms_norm, raising=False)
    if tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) < (2, 5):
        sdpa = F.scaled_dot_product_attention
        def attention(q, k, v, *, enable_gqa=False, **kwargs):
            if enable_gqa:
                k = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
                v = v.repeat_interleave(q.size(1) // v.size(1), dim=1)
            return sdpa(q, k, v, **kwargs)
        monkeypatch.setattr(F, 'scaled_dot_product_attention', attention)
    import nanochat.optim as optim
    monkeypatch.setattr(optim, 'adamw_step_fused', optim.adamw_step_fused.__wrapped__)
    monkeypatch.setattr(optim, 'muon_step_fused', optim.muon_step_fused.__wrapped__)


def make_model(**kwargs):
    torch.manual_seed(17)
    values = dict(sequence_len=8, vocab_size=16, n_layer=2, n_embd=32,
                  n_head=2, n_kv_head=1, action_dim=8, window_pattern='L')
    values.update(kwargs)
    model = LLWM(LLWMConfig(**values))
    model.init_weights()
    return model


def batch():
    tokens = torch.tensor([[1,2,3,4,5], [6,7,8,9,10]])
    return tokens[:,:-1], tokens[:,1:]


@pytest.mark.parametrize('depth,predictor', [(1,1),(2,2),(3,3),(4,3),(8,6),(12,9),(20,15)])
def test_reference_depth_ratio(depth, predictor):
    config = LLWMConfig(n_layer=depth)
    assert config.predictor_layers == predictor
    assert LLWMConfig(n_layer=depth, predictor_layers=2).predictor_layers == 2
    assert config_from_dict(asdict(config)) == config


@pytest.mark.parametrize('kwargs', [dict(n_layer=0),dict(predictor_layers=0),dict(action_dim=0),
                                   dict(action_layers=0),dict(target_kl=-1),dict(kl_lambda=float('nan'))])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        LLWMConfig(**kwargs)


def test_action_encoder_halving():
    encoder = ActionEncoder(256,32,4)
    assert [(m.in_features,m.out_features) for m in encoder.trunk if isinstance(m,torch.nn.Linear)] == [
        (512,256),(256,128),(128,64),(64,32)]
    assert encoder(torch.randn(2,3,512)).shape == (2,3,64)


def test_reference_losses_shifted_memory_and_gradients():
    model = make_model(target_kl=0.2,kl_lambda=0.3,ce_weight=0.7,memory_weight=0.4,memory_loss='mse')
    idx,targets = batch()
    seen = []
    hook = model.predictor_norm.register_forward_hook(lambda module,args,out: seen.append(out))
    values = model(idx,targets,return_loss_components=True)
    hook.remove()
    with torch.no_grad():
        next_memory = model.encode(torch.cat((idx,targets[:,-1:]),dim=1))[:,1:]
    torch.testing.assert_close(values['memory_loss'],F.mse_loss(seen[0].float(),next_memory.float()))
    expected = 0.7*values['ce'] + 0.4*values['memory_loss'] + 0.3*F.relu(values['kl_total']-0.2).square()
    torch.testing.assert_close(values['loss'],expected)
    assert set(values) == {'loss','ce','memory_loss','kl','kl_total','kl_excess','kl_penalty'}
    values['loss'].backward()
    for name,p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    for module in (model.compressor,model.action_encoder,model.predictor,model.llm_head):
        assert any(p.grad.abs().sum() > 0 for p in module.parameters())


def test_posterior_detaches_memories():
    model = make_model()
    memory = torch.randn(2,3,32,requires_grad=True)
    target = torch.randn_like(memory,requires_grad=True)
    mean,log_std = model.posterior(memory,target)
    (mean.square().mean()+log_std.square().mean()).backward()
    assert memory.grad is None and target.grad is None
    assert all(p.grad is not None for p in model.action_encoder.parameters())


def test_no_whole_predictor_residual():
    model = make_model()
    with torch.no_grad():
        model.predictor_norm.weight.zero_()
        model.predictor_norm.bias.fill_(0.25)
    actual = model.predict_features(torch.randn(2,3,32),torch.randn(2,3,8))
    torch.testing.assert_close(actual,torch.full_like(actual,0.25))


def test_all_ignored_targets_return_finite_zero():
    model = make_model()
    idx,targets = batch()
    targets = torch.full_like(targets,-1)
    values = model(idx,targets,return_loss_components=True)
    assert all(value.item() == 0 for value in values.values())
    values['loss'].backward()
    model.eval()
    assert model(idx,targets).item() == 0


def test_eval_and_rl_use_causal_prior():
    model = make_model().eval()
    idx,targets = batch()
    logits = model(idx)
    changed = idx.clone(); changed[:,2:] = (changed[:,2:]+1)%16
    torch.testing.assert_close(model(changed)[:,:2],logits[:,:2])
    expected = F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none')
    torch.testing.assert_close(model(idx,targets,loss_reduction='none'),expected)
    torch.testing.assert_close(model(idx,targets),expected.mean())
    model.train()
    torch.testing.assert_close(model(idx,targets,loss_reduction='none'),expected)


def test_optimizer_coverage_and_skipping_posterior_for_rl():
    model = make_model()
    optimizer = model.setup_optimizer()
    params = [p for g in optimizer.param_groups for p in g['params']]
    assert len(params) == len(set(params)) == len(list(model.parameters()))
    counts = model.num_scaling_params()
    assert sum(v for k,v in counts.items() if k!='total') == counts['total']
    idx,targets = batch()
    before = {n:p.detach().clone() for n,p in model.action_encoder.named_parameters()}
    model(idx,targets,loss_reduction='none').mean().backward()
    optimizer.step()
    for n,p in model.action_encoder.named_parameters():
        torch.testing.assert_close(p,before[n]); assert p not in optimizer.state
    optimizer.zero_grad(set_to_none=True)
    model(idx,targets).backward(); optimizer.step()
    assert any(not torch.equal(p,before[n]) for n,p in model.action_encoder.named_parameters())


def test_meta_checkpoint_roundtrip_and_legacy_factory():
    original = make_model()
    with torch.device('meta'):
        restored = create_model(config_from_dict(asdict(original.config)))
    restored.to_empty(device='cpu'); restored.init_weights()
    buffer = io.BytesIO(); torch.save(original.state_dict(),buffer); buffer.seek(0)
    restored.load_state_dict(torch.load(buffer,weights_only=True))
    original.eval(); restored.eval(); idx,_ = batch()
    torch.testing.assert_close(original(idx),restored(idx))
    legacy = config_from_dict(dict(sequence_len=8,vocab_size=16,n_layer=2,n_embd=32,n_head=2,n_kv_head=1,window_pattern='L'))
    assert type(create_model(legacy)).__name__ == 'GPT'


def test_independent_gaussian_initial_actions_and_planning():
    from nanochat.planning import plan, sample_actions
    actions = sample_actions(2,5,8,device='cpu',seed=123)
    expected = torch.randn(2,5,8,generator=torch.Generator().manual_seed(123))
    torch.testing.assert_close(actions,expected)
    assert not torch.equal(actions[:,0],actions[:,1])
    model = make_model()
    weights = {n:p.detach().clone() for n,p in model.named_parameters()}
    result = plan(model,torch.tensor([[1,2,3]]),torch.tensor([4]),horizon=3,iterations=3)
    assert result['tokens'].shape == (1,3)
    assert result['best_loss'] < result['history'][0]['loss']
    assert model.training
    for n,p in model.named_parameters():
        assert p.requires_grad and p.grad is None
        torch.testing.assert_close(p,weights[n])


def test_cached_predictions_and_single_token_generation():
    from nanochat.engine import KVCache
    from nanochat.common import COMPUTE_DTYPE
    model = make_model()
    idx, targets = batch()
    optimizer = model.setup_optimizer()
    for _ in range(2):
        model(idx, targets).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    model.eval()
    idx = idx[:1]
    cache = KVCache(batch_size=1, seq_len=8, num_heads=1, head_dim=16,
                    num_layers=2, device=torch.device('cpu'), dtype=COMPUTE_DTYPE)
    expected = model(idx)
    pieces = [model(idx[:, :2], kv_cache=cache)]
    pieces.extend(model(idx[:, i:i+1], kv_cache=cache) for i in range(2, idx.size(1)))
    tolerance = 0.01 if COMPUTE_DTYPE == torch.bfloat16 else 1e-6
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, atol=tolerance, rtol=tolerance * 10)
    assert cache.get_pos() == idx.size(1)
    assert len(list(model.generate([1], max_tokens=2, temperature=0))) == 2


def test_native_checkpoint_loading_and_optimizer_resume(tmp_path, monkeypatch):
    from nanochat import checkpoint_manager as checkpoints
    class Tokenizer:
        def get_vocab_size(self):
            return 16
    monkeypatch.setattr(checkpoints, 'get_tokenizer', lambda: Tokenizer())
    model = make_model()
    optimizer = model.setup_optimizer()
    idx, targets = batch()
    model(idx, targets).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    expected = model(idx)
    checkpoints.save_checkpoint(str(tmp_path), 1, model.state_dict(), optimizer.state_dict(),
                                {'model_config': asdict(model.config)})
    restored, _, metadata = checkpoints.build_model(str(tmp_path), 1, torch.device('cpu'), 'eval')
    assert isinstance(restored, LLWM) and metadata['model_config']['predictor_layers'] == 2
    torch.testing.assert_close(restored(idx), expected)
    _, state, _ = checkpoints.load_checkpoint(str(tmp_path), 1, torch.device('cpu'), load_optimizer=True)
    restored_optimizer = restored.setup_optimizer()
    restored_optimizer.load_state_dict(state)
    model.train()
    restored.train()
    rng = torch.get_rng_state()
    model(idx, targets).backward()
    optimizer.step()
    torch.set_rng_state(rng)
    restored(idx, targets).backward()
    restored_optimizer.step()
    for original, copy in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(original, copy)


def test_pretraining_entrypoint_defaults_to_llwm_and_resumes(tmp_path, monkeypatch):
    import gc
    import json
    import runpy
    import sys
    from types import ModuleType
    from nanochat import common, tokenizer
    class Tokenizer:
        def get_vocab_size(self):
            return 16
        def encode(self, text):
            return [3] if text == '.' else [1, 2]
        def __call__(self, text, prepend=None):
            return ([0] if prepend is not None else []) + self.encode(text)
        def decode(self, tokens):
            return 'synthetic planned sample'
    def batches(tokenizer, batch_size, length, **kwargs):
        while True:
            tokens = torch.arange(batch_size*(length+1)).view(batch_size,length+1) % 16
            yield tokens[:,:-1].contiguous(), tokens[:,1:].contiguous(), dict(pq_idx=0,rg_idx=0,epoch=1)
    monkeypatch.setenv('NANOCHAT_BASE_DIR', str(tmp_path))
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    monkeypatch.setenv('PYTORCH_ALLOC_CONF', '')
    monkeypatch.setattr(tokenizer, 'get_tokenizer', lambda: Tokenizer())
    monkeypatch.setattr(tokenizer, 'get_token_bytes', lambda device: torch.ones(16, dtype=torch.long, device=device))
    dataloader = ModuleType('nanochat.dataloader')
    dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit = batches
    dataloader.tokenizing_distributed_data_loader_bos_bestfit = batches
    monkeypatch.setitem(sys.modules, 'nanochat.dataloader', dataloader)
    monkeypatch.setattr(common, 'compile_model', lambda model, **kwargs: model)
    for name in ('freeze', 'disable', 'collect'):
        monkeypatch.setattr(gc, name, lambda: None)
    args = ['scripts.base_train', '--device-type=cpu', '--depth=2', '--aspect-ratio=16',
            '--head-dim=16', '--max-seq-len=4', '--action-dim=8', '--device-batch-size=1',
            '--total-batch-size=4', '--eval-every=-1', '--sample-every=-1',
            '--core-metric-every=-1', '--warmup-steps=0', '--window-pattern=L']
    monkeypatch.setattr(sys, 'argv', args + ['--num-iterations=1'])
    runpy.run_module('scripts.base_train', run_name='__main__')
    folder = tmp_path/'base_checkpoints'/'llwm_d2'
    with (folder/'meta_000001.json').open() as f:
        metadata = json.load(f)
    assert metadata['model_config']['architecture'] == 'llwm'
    assert metadata['model_config']['predictor_layers'] == 2
    monkeypatch.setattr(sys, 'argv', args + ['--num-iterations=2', '--resume-from-step=1'])
    runpy.run_module('scripts.base_train', run_name='__main__')
    assert (folder/'model_000002.pt').is_file()
    assert (folder/'optim_000002_rank0.pt').is_file()

    # Enable real planning at the periodic CORE/sample call sites on resume.
    import scripts.base_eval as base_eval
    from nanochat import planning_eval
    actual_plan = planning_eval.plan
    plans = []
    def recorded_plan(model, prompt, goal, **kwargs):
        plans.append((goal.tolist(), kwargs['horizon']))
        return actual_plan(model, prompt, goal, **kwargs)
    monkeypatch.setattr(planning_eval, 'plan', recorded_plan)
    def core(model, tokenizer, device, max_per_task, planning, max_new_tokens):
        planning_eval.planned_completion(model, tokenizer, [0, 1], max_new_tokens, planning)
        return dict(core_metric=0.25, centered_results={}, eval_mode='plan',
                    planning_goal=planning.goal, planning_horizon=max_new_tokens)
    monkeypatch.setattr(base_eval, 'evaluate_core', core)
    monkeypatch.setattr(sys, 'argv', args + ['--num-iterations=3', '--resume-from-step=2',
        '--core-metric-every=1', '--sample-every=1', '--eval-mode=plan',
        '--eval-max-new-tokens=3', '--plan-iterations=1', '--plan-goal=.'])
    runpy.run_module('scripts.base_train', run_name='__main__')
    assert len(plans) == 16  # CORE + seven samples at step 2 and final step 3.
    assert all(goal == [3] and horizon == 3 for goal, horizon in plans)
    with (folder/'meta_000003.json').open() as f:
        assert json.load(f)['core_results']['planning_goal'] == '.'
