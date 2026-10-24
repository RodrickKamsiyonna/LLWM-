"""Planning budgets, answer isolation, CORE scoring and launch recipes."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from test_llwm import portable_torch, make_model
from nanochat import planning_eval as pe
from nanochat.core_eval import evaluate_example


class Tokenizer:
    ids = {'a':1, 'b':2, '.':3, 'q':4, ' ':5, '\n':6}
    def get_bos_token_id(self):
        return 0
    def encode(self, text):
        return [self.ids[ch] for ch in text]
    def __call__(self, text, prepend=None):
        prefix = [] if prepend is None else [prepend]
        if isinstance(text, list):
            return [prefix+self.encode(value) for value in text]
        return prefix+self.encode(text)


@pytest.fixture
def fake_plan(monkeypatch):
    calls = []
    def run(model, prompt, goal, **kwargs):
        calls.append((prompt.clone(), goal.clone(), dict(kwargs)))
        horizon = kwargs['horizon']
        logits = torch.zeros(prompt.size(0), horizon, 8)
        logits[..., 1] = 4  # 'a', regardless of the benchmark's gold label.
        return {'logits':logits, 'tokens':logits.argmax(-1)}
    monkeypatch.setattr(pe, 'plan', run)
    return calls


@pytest.fixture
def model():
    return SimpleNamespace(config=SimpleNamespace(architecture='llwm'), get_device=lambda:torch.device('cpu'))


def test_defaults_and_architecture_selection(model):
    parser = argparse.ArgumentParser()
    pe.add_planning_args(parser)
    args = parser.parse_args([])
    assert args.eval_max_new_tokens == 16
    options = pe.planning_for_model(model,args)
    assert options.goal == '.' and options.iterations == 100
    args.eval_mode='prior'
    assert pe.planning_for_model(model,args) is None
    legacy = SimpleNamespace(config=SimpleNamespace())
    assert pe.planning_for_model(legacy) is None
    args.eval_mode='plan'
    with pytest.raises(ValueError,match='LLWM'):
        pe.planning_for_model(legacy,args)


def test_completion_uses_period_and_exact_output_budget(model,fake_plan):
    options = pe.PlanningOptions(iterations=2)
    result = pe.planned_completion(model,Tokenizer(),[0,4],7,options,num_samples=3)
    prompt,goal,settings=fake_plan[0]
    assert prompt.shape == (3,2)
    assert goal.tolist() == [3]
    assert settings['horizon'] == 7
    assert settings['return_logits'] is True
    assert result['tokens'].shape == (3,7)
    with pytest.raises(ValueError,match='goal token count'):
        pe.planned_completion(model,Tokenizer(),[0,4],1,replace(options,goal='..'))


def test_candidate_scoring_reuses_plans_and_never_optimizes_answer(model,fake_plan):
    scores,exact=pe.planned_continuation_scores(model,Tokenizer(),[[0,4,1,1,1,1,1],[0,4,2]],
                                              [2,2],[7,3],3,pe.PlanningOptions())
    assert scores[0] < scores[1] and exact == [True,False]
    # The first plan is shared. The next block sees only the completed prefix.
    assert len(fake_plan) == 2
    assert fake_plan[0][0].tolist() == [[0,4]]
    assert fake_plan[1][0].tolist() == [[0,4,1,1,1]]
    assert all(goal.tolist()==[3] and settings['horizon']==3 for _,goal,settings in fake_plan)


@pytest.mark.parametrize('kind,item',[
    ('multiple_choice',dict(query='q',choices=['a','b'],gold=0)),
    ('schema',dict(context_options=['q','b'],continuation='a',gold=0)),
    ('language_modeling',dict(context='q',continuation='a')),
])
def test_core_tasks_use_planning_under_no_grad(model,fake_plan,kind,item):
    meta=dict(task_type=kind,num_fewshot=0,continuation_delimiter='')
    result=evaluate_example(0,model,Tokenizer(),[item],torch.device('cpu'),meta,
                            planning=pe.PlanningOptions(),max_new_tokens=4)
    assert result
    assert fake_plan and all(settings['horizon']==4 for _,_,settings in fake_plan)
    if 'gold' in item:
        before=[(p.tolist(),g.tolist()) for p,g,_ in fake_plan]
        fake_plan.clear()
        changed=dict(item,gold=1)
        assert not evaluate_example(0,model,Tokenizer(),[changed],torch.device('cpu'),meta,
                                    planning=pe.PlanningOptions(),max_new_tokens=4)
        assert [(p.tolist(),g.tolist()) for p,g,_ in fake_plan] == before


def test_real_planning_can_optimize_actions_inside_eval_no_grad():
    model=make_model()
    weights={name:p.detach().clone() for name,p in model.named_parameters()}
    with torch.no_grad():
        result=pe.planned_completion(model,Tokenizer(),[0,4],3,pe.PlanningOptions(iterations=2))
    assert result['logits'].shape == (1,3,16)
    assert not result['logits'].requires_grad
    assert result['best_loss'] < result['history'][0]['loss']
    assert model.training
    for name,p in model.named_parameters():
        assert p.requires_grad and p.grad is None
        torch.testing.assert_close(p,weights[name])


def test_limited_core_questions_keep_full_fewshot_pool(model, fake_plan):
    pool = [dict(query='q', choices=['a', 'b'], gold=0) for _ in range(30)]
    meta = dict(task_type='multiple_choice', num_fewshot=25,
                continuation_delimiter='', fewshot_pool=pool)
    assert evaluate_example(0, model, Tokenizer(), pool[:1], torch.device('cpu'), meta,
                            planning=pe.PlanningOptions(), max_new_tokens=4)


@pytest.mark.parametrize('kind', ['categorical', 'generative'])
def test_chatcore_uses_planning_and_its_allowed_token_budget(monkeypatch, model, fake_plan, kind):
    # Import evaluation without downloading task datasets or importing Parquet.
    import importlib.util
    from types import ModuleType
    for name, cls in [('humaneval', 'HumanEval'), ('mmlu', 'MMLU'), ('arc', 'ARC'), ('gsm8k', 'GSM8K')]:
        module = ModuleType(f'tasks.{name}')
        setattr(module, cls, object)
        monkeypatch.setitem(sys.modules, f'tasks.{name}', module)
    path = Path(__file__).resolve().parents[1]/'scripts'/'chat_eval.py'
    spec = importlib.util.spec_from_file_location('_chat_planning_test', path)
    chat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chat)
    class Task:
        eval_type = kind
        def __init__(self, **kwargs): pass
        def __len__(self): return 1
        def __getitem__(self, idx): return dict(letters=['a', 'b'])
        def evaluate(self, conversation, answer): return answer.startswith('a')
    tokenizer = Tokenizer()
    tokenizer.render_for_completion = lambda conversation: [0, 4]
    tokenizer.decode = lambda ids: ''.join('a' if i == 1 else '?' for i in ids)
    monkeypatch.setattr(chat, 'ARC', Task)
    score = chat.run_chat_eval('ARC-Easy', model, tokenizer, None,
                               max_new_tokens=9, planning=pe.PlanningOptions())
    assert score == 1.0
    assert len(fake_plan) == 1
    assert fake_plan[0][1].tolist() == [3] and fake_plan[0][2]['horizon'] == 9


def test_sample_adapter_returns_budgeted_rollout(model,fake_plan):
    engine=pe.PlanningEngine(model,Tokenizer(),pe.PlanningOptions())
    sequences,masks=engine.generate_batch([0,4],num_samples=2,max_tokens=5)
    assert sequences == [[0,4]+[1]*5]*2
    assert masks == [[0,0]+[1]*5]*2
    assert fake_plan[0][2]['horizon'] == 5


@pytest.mark.parametrize('script',['speedrun.sh','miniseries.sh','scaling_laws.sh','runcpu.sh'])
def test_run_recipes_preview_without_training(tmp_path,script):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,DRY_RUN='1',SKIP_DATA='1',NANOCHAT_BASE_DIR=str(tmp_path),
             DEPTHS='4',FLOPS_BUDGETS='1e18',EVAL_MAX_NEW_TOKENS='7',PLAN_ITERATIONS='2',PLAN_GOAL='.')
    env.pop('NANOCHAT_DEVICE',None)
    result=subprocess.run(['bash',str(root/'runs'/script)],env=env,capture_output=True,text=True,check=True)
    assert '--model-type=llwm' in result.stdout
    assert '--plan-goal=.' in result.stdout and '--eval-max-new-tokens=7' in result.stdout
    assert '--fp8' not in result.stdout
    if script=='runcpu.sh':
        assert '--device-type=cpu' in result.stdout and 'torchrun' not in result.stdout
    else:
        assert 'scripts.xla_launch scripts.base_train' in result.stdout
    assert not list(tmp_path.iterdir())


def test_sweep_summary_uses_actual_checkpoint_batch_and_planning_metadata(tmp_path):
    folder=tmp_path/'base_checkpoints'/'test'
    folder.mkdir(parents=True)
    meta=dict(model_config=dict(n_layer=4,n_embd=256,predictor_layers=3),step=5,total_batch_size=64,
              parameter_counts=dict(total=1000,action_encoder=100,predictor=200),val_bpb=1.2,
              core_results=dict(core_metric=0.3,eval_mode='plan',planning_goal='.',planning_horizon=7))
    (folder/'meta_000005.json').write_text(json.dumps(meta))
    output=tmp_path/'results.csv'
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,'-m','scripts.run_summary','--model-tag=test',f'--results-file={output}'],
                   cwd=root,env=dict(os.environ,NANOCHAT_BASE_DIR=str(tmp_path)),capture_output=True,check=True)
    import csv
    with output.open() as f:
        row=next(csv.DictReader(f))
    assert row['tokens_trained']=='320' and row['core_mode']=='plan' and row['planning_horizon']=='7'
