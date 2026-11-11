import torch
from nanochat.llwm import LLWM, LLWMConfig
from nanochat.planning import mpc_plan


def make_model():
    torch.manual_seed(9)
    m = LLWM(LLWMConfig(sequence_len=16,vocab_size=16,n_layer=2,n_embd=32,n_head=2,
        n_kv_head=1,window_pattern='L',action_dim=8,action_layers=2,predictor_layers=6,
        prediction_horizon=1,head_source='predictor',head_training='joint',
        action_conditioning='next_token',planning_mode='straight_through',memory_weight=0.,
        token_objective='embedding_mse',kl_lambda=3.))
    m.init_weights()
    return m


def test_masked_embedding_loss_gradients_and_optimizer_coverage():
    m=make_model();x=torch.tensor([[1,2,3]]);y=torch.tensor([[2,3,-1]])
    predicted=[]
    hook=m.embedding_head.register_forward_hook(lambda mod,args,out:predicted.append(out))
    loss=m(x,y,return_loss_components=True);hook.remove()
    target=m.transformer.wte(y.clamp(min=0)).detach().float()
    expected=(predicted[0].float()[:,:2]-target[:,:2]).square().mean()
    torch.testing.assert_close(loss['embedding_mse'],expected)
    torch.testing.assert_close(loss['loss'],expected+loss['kl_penalty'])
    loss['loss'].backward()
    for module in (m.embedding_head,m.llm_head_mlp,m.predictor,m.action_encoder,m.compressor):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    opt=m.setup_optimizer();grouped=[p for g in opt.param_groups for p in g['params']]
    assert len(grouped)==len(set(grouped))==len(list(m.parameters()))
    for group in opt.param_groups:
        flags=[p.grad is not None for p in group['params']]
        assert not any(flags) or all(flags), 'partially active optimizer group'


def test_nearest_embedding_decoding():
    m=make_model();e=torch.randn(2,3,32)
    distances=(e[...,None,:]-m.transformer.wte.weight[:16]).square().mean(-1)
    torch.testing.assert_close(m.embedding_logits(e).argmax(-1),distances.argmin(-1))


def test_stagewise_embedding_gradient_and_two_round_mpc():
    from nanochat.engine import DifferentiableKVCache
    from nanochat.st_planning import st_rollout
    m=make_model().eval()
    for p in m.parameters():p.requires_grad_(False)
    cache=DifferentiableKVCache(2,8)
    h=m.encode(torch.tensor([[1,2]]),kv_cache=cache)[:,-1].detach()
    a=torch.randn(1,3,8,requires_grad=True)
    full=st_rollout(m,h,cache,a,checkpoint_steps=False,stagewise=False)
    gf,=torch.autograd.grad(full.square().mean(),a)
    staged=st_rollout(m,h,cache,a,stagewise=True)
    gs,=torch.autograd.grad(staged.square().mean(),a)
    torch.testing.assert_close(staged,full)
    torch.testing.assert_close(gs,gf,atol=2e-5,rtol=2e-4)
    assert gf[:,0].abs().sum()>0
    r=mpc_plan(m,torch.tensor([[1,2]]),torch.tensor([4]),horizon=4,execute_steps=2,iterations=2,lr=.001)
    assert r['tokens'].shape==(1,4) and r['goal_objective']=='goal_mse'
    assert all(x['goal_mse']<=x['initial_loss'] for x in r['round_history'])


def test_future_token_teacher_does_not_update_embedding_table():
    m = make_model()
    # Distinct IDs distinguish context gradients from future-teacher gradients.
    x, y = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])
    m(x, y, return_loss_components=True)['loss'].backward()
    grad = m.transformer.wte.weight.grad
    assert grad is not None and grad[1:4].abs().sum() > 0
    assert grad[4:7].eq(0).all()


def test_action_posterior_stops_both_observed_inputs():
    m = make_model()
    memory = torch.randn(1, 3, 32, requires_grad=True)
    mean, log_std = m.token_posterior(memory, torch.tensor([[4, 5, 6]]))
    (mean.square().mean() + log_std.square().mean()).backward()
    assert memory.grad is None
    assert m.transformer.wte.weight.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.action_encoder.parameters())


def test_trainable_mse_target_receives_exact_target_gradient():
    m = make_model()
    m.config.detach_embedding_target = False
    x, y = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])
    predictions = []
    hook = m.embedding_head.register_forward_hook(lambda mod, args, out: predictions.append(out))
    loss = m(x, y, return_loss_components=True)
    hook.remove()
    target = m.transformer.wte(y).detach().float()
    expected = 2 * (target - predictions[0].detach().float()) / target.numel()
    loss['loss'].backward()
    torch.testing.assert_close(m.transformer.wte.weight.grad[y[0]], expected[0])
    assert m.transformer.wte.weight.grad[y[0]].abs().sum() > 0
    # Enabling the MSE target gradient does not enable posterior-input gradients.
    m.zero_grad(set_to_none=True)
    memory = torch.randn(1, 3, 32, requires_grad=True)
    mean, log_std = m.token_posterior(memory, y)
    (mean.square().mean() + log_std.square().mean()).backward()
    assert memory.grad is None and m.transformer.wte.weight.grad is None
