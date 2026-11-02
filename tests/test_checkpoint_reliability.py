import json
import random
import numpy as np
import pytest
import torch
from test_llwm import portable_torch,make_model,batch
from nanochat.reliable_checkpoint import TrainingCheckpoints,read_complete,capture_rng,restore_rng,complete_steps
from nanochat.distributed_utils import accumulation_steps


def manager(path,contract=None,keep=1):
    return TrainingCheckpoints(path,'test',0,1,torch.device('cpu'),contract or {'schedule':6000},keep=keep)


@pytest.mark.parametrize('tokens,steps',[(16384,2),(131072,16)])
def test_original_batch_accumulation(tokens,steps):
    assert accumulation_steps(tokens,1,1024,8)==steps
    with pytest.raises(ValueError):accumulation_steps(tokens+1,1,1024,8)


def test_full_state_atomic_checkpoint_and_rng_restore(tmp_path):
    model=make_model();optimizer=model.setup_optimizer();idx,targets=batch()
    model(idx,targets).backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    checkpoint=manager(tmp_path);checkpoint.verify_storage()
    rng=capture_rng(torch.device('cpu'))
    checkpoint.save(1,model.state_dict(),optimizer.state_dict(),{'step':1},dict(rng=rng,loader={'prefetched':idx}))
    expected=(random.random(),np.random.rand(),torch.rand(3))
    weights,state,meta,runtime,manifest=checkpoint.load(1)
    restored=make_model();restored.load_state_dict(weights);restored_opt=restored.setup_optimizer();restored_opt.load_state_dict(state)
    restore_rng(runtime['rng'],torch.device('cpu'))
    assert random.random()==expected[0] and np.random.rand()==expected[1]
    torch.testing.assert_close(torch.rand(3),expected[2])
    reference_rng=torch.get_rng_state()
    model(idx,targets).backward();optimizer.step()
    torch.set_rng_state(reference_rng)
    restored(idx,targets).backward();restored_opt.step()
    for left,right in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(left,right)
    assert manifest['optimizer_kind']=='replicated'


def test_incomplete_mismatched_and_corrupt_checkpoints_are_rejected(tmp_path):
    checkpoint=manager(tmp_path)
    checkpoint.save(25,{'p':torch.ones(2)},{'state':{}},{'step':25},{'loader':{}})
    with pytest.raises(ValueError,match='missing'):read_complete(tmp_path,50)
    with pytest.raises(ValueError,match='world size'):read_complete(tmp_path,25,load_optimizer=True,expected_world=8)
    with pytest.raises(ValueError,match='contract'):read_complete(tmp_path,25,load_optimizer=True,contract={'schedule':300})
    manifest=json.loads((tmp_path/'complete_000025.json').read_text())
    (tmp_path/manifest['model']['name']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):read_complete(tmp_path,25)


def test_completed_schedule_can_be_extended_only_without_other_contract_changes(tmp_path):
    original={'num_iterations':6000,'model':'llwm','optimizer':{'kl_lambda':2.0},'scheduler':{'warmdown_ratio':0.65}}
    checkpoint=manager(tmp_path,contract=original)
    checkpoint.save(6000,{'p':torch.ones(2)},{'state':{}},{'step':6000},{'loader':{}})
    extended={**original,'num_iterations':12000}
    restored=read_complete(tmp_path,6000,load_optimizer=True,expected_world=1,contract=extended,
                           allow_horizon_extension=True)
    assert restored[2]['step']==6000
    with pytest.raises(ValueError,match='contract'):
        read_complete(tmp_path,6000,load_optimizer=True,expected_world=1,
                      contract={**extended,'optimizer':{'kl_lambda':2.5}},allow_horizon_extension=True)
    with pytest.raises(ValueError,match='contract'):
        read_complete(tmp_path,6000,load_optimizer=True,expected_world=1,
                      contract={**original,'num_iterations':6000},allow_horizon_extension=True)


def test_retention_keeps_only_latest_complete_and_preserves_legacy_files(tmp_path):
    legacy=tmp_path/'model_000001.pt';legacy.write_bytes(b'legacy preserved')
    checkpoint=manager(tmp_path,keep=1)
    for step in (25,50,75):checkpoint.save(step,{'p':torch.ones(2)},{'state':{}},{'step':step},{'loader':{}})
    assert complete_steps(tmp_path)==[75]
    assert checkpoint.latest()==75 and legacy.read_bytes()==b'legacy preserved'


def test_no_completion_manifest_if_an_artifact_write_fails(tmp_path,monkeypatch):
    checkpoint=manager(tmp_path)
    original=checkpoint.store.put_file
    def fail(name,source,exclusive=False):
        if 'optim_' in name:raise OSError('injected storage failure')
        return original(name,source,exclusive)
    monkeypatch.setattr(checkpoint.store,'put_file',fail)
    with pytest.raises(RuntimeError,match='transaction'):checkpoint.save(25,{'p':torch.ones(2)},{'state':{}},{'step':25},{})
    assert complete_steps(tmp_path)==[]


def test_failed_new_checkpoint_preserves_previous_latest(tmp_path,monkeypatch):
    checkpoint=manager(tmp_path)
    checkpoint.save(25,{'p':torch.ones(2)},{'state':{}},{'step':25},{})
    original=checkpoint.store.put_file
    def fail(name,source,exclusive=False):
        if 'optim_' in name:raise OSError('injected storage failure')
        return original(name,source,exclusive)
    monkeypatch.setattr(checkpoint.store,'put_file',fail)
    with pytest.raises(RuntimeError,match='transaction'):checkpoint.save(50,{'p':torch.zeros(2)},{'state':{}},{'step':50},{})
    assert complete_steps(tmp_path)==[25]
