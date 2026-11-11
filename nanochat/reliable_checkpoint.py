"""Transactional, topology-aware training checkpoints on shared storage.

XLA MuonAdamW is fully replicated: save one optimizer replica, plus every rank's
RNG/input-stream state. CUDA's sharded optimizer still saves every rank's shard.
Only complete manifests are eligible for exact resume or retention.
"""
import json
import os
from pathlib import Path
import random
import socket
import tempfile
import time
import uuid
import numpy as np
import torch

from nanochat.storage import ArtifactStore, file_record
from nanochat.distributed_utils import gather_json, require_equal
from nanochat import diagnostics as diag

FORMAT=1


def capture_rng(device):
    state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
    if device.type=='cuda': state['cuda']=torch.cuda.get_rng_state(device)
    if device.type=='xla':
        import torch_xla.core.xla_model as xm
        state['xla']=xm.get_rng_state(device)
    return state


def restore_rng(state,device):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        if device.type!='cuda': raise ValueError('CUDA RNG state cannot be exactly resumed on another backend')
        torch.cuda.set_rng_state(state['cuda'],device)
    if 'xla' in state:
        if device.type!='xla': raise ValueError('XLA RNG state cannot be exactly resumed on another backend')
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(state['xla'],device)


def cpu_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    return value


def complete_steps(root):
    store=ArtifactStore(root)
    result=[]
    for name in store.list('complete_'):
        if '/' not in name and name.endswith('.json'):
            try: step=int(name[len('complete_'):-5])
            except ValueError: continue
            manifest=store.get_json(name)
            if manifest.get('format')==FORMAT and manifest.get('step')==step:
                result.append(step)
    return sorted(result)


def _read_file(store,record,cache=None):
    if cache:
        from filelock import FileLock
        cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
        path=cache/record['sha256']
        with FileLock(str(path)+'.lock'):
            if not path.exists() or file_record(path,record['name'])!=record:
                temporary=cache/(record['sha256']+'.tmp-'+uuid.uuid4().hex)
                try:
                    store.get_file(record['name'],temporary)
                    if file_record(temporary,record['name'])!=record:
                        raise ValueError(f'Checkpoint checksum/size mismatch: {record["name"]}')
                    os.replace(temporary,path)
                finally:
                    temporary.unlink(missing_ok=True)
            # Trusted own checkpoints, checked against the completion manifest.
            return torch.load(path,map_location='cpu',weights_only=False)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'artifact.pt'
        store.get_file(record['name'],path)
        if file_record(path,record['name'])!=record:
            raise ValueError(f'Checkpoint checksum/size mismatch: {record["name"]}')
        return torch.load(path,map_location='cpu',weights_only=False)


def _is_horizon_extension(checkpoint_contract,requested_contract,step):
    """Allow an explicit longer budget, preserving every other contract field."""
    if not isinstance(checkpoint_contract,dict) or not isinstance(requested_contract,dict):
        return False
    old_horizon=checkpoint_contract.get('num_iterations')
    new_horizon=requested_contract.get('num_iterations')
    if not isinstance(old_horizon,int) or not isinstance(new_horizon,int):
        return False
    if not 0 <= step <= old_horizon or new_horizon<=old_horizon:
        return False
    old=dict(checkpoint_contract); new=dict(requested_contract)
    old.pop('num_iterations'); new.pop('num_iterations')
    return old==new


def read_complete(root,step,rank=0,load_optimizer=False,expected_world=None,contract=None,cache=None,load_model=True,
                  allow_horizon_extension=False):
    store=ArtifactStore(root)
    name=f'complete_{step:06d}.json'
    if not store.exists(name): raise ValueError(f'Incomplete or missing checkpoint: {root}/{name}')
    manifest=store.get_json(name)
    if manifest.get('format')!=FORMAT or manifest.get('step')!=step:
        raise ValueError('Unsupported/mismatched completion manifest')
    world=manifest['world_size']
    if len(manifest['runtime'])!=world or set(manifest['runtime'])!={str(i) for i in range(world)}:
        raise ValueError('Checkpoint manifest lacks rank runtime files')
    if load_optimizer and expected_world is not None and expected_world!=world:
        raise ValueError(f'Exact resume requires original world size {world}, got {expected_world}')
    contract_matches=(contract is None or manifest['contract']==contract)
    if (load_optimizer and not contract_matches and allow_horizon_extension
            and _is_horizon_extension(manifest['contract'],contract,step)):
        contract_matches=True
    if load_optimizer and not contract_matches:
        raise ValueError('Training contract mismatch: architecture, batch, objective, scheduler horizon or optimizer settings changed')
    records=[manifest['model'],manifest['metadata'],*manifest['runtime'].values(),*manifest['optimizer'].values()]
    if any(not store.exists(item['name']) for item in records):
        raise ValueError('Completion manifest references missing checkpoint files')
    with tempfile.TemporaryDirectory() as tmp:
        metadata_path=Path(tmp)/'meta.json'
        store.get_file(manifest['metadata']['name'],metadata_path)
        if file_record(metadata_path,manifest['metadata']['name'])!=manifest['metadata']:
            raise ValueError('Checkpoint metadata checksum mismatch')
        metadata=json.loads(metadata_path.read_text())
    model=_read_file(store,manifest['model'],cache) if load_model else None
    optimizer=runtime=None
    if load_optimizer:
        if not 0<=rank<world: raise ValueError('Rank outside checkpoint topology')
        key='0' if manifest['optimizer_kind']=='replicated' else str(rank)
        optimizer=_read_file(store,manifest['optimizer'][key],cache)
        runtime=_read_file(store,manifest['runtime'][str(rank)],cache)
        if runtime['rank']!=rank or runtime['step']!=step:
            raise ValueError('Checkpoint rank/step runtime mismatch')
    return model,optimizer,metadata,runtime,manifest


class TrainingCheckpoints:
    def __init__(self,root,run_id,rank,world_size,device,contract,keep=1,shared=False,cache=None,
                 allow_horizon_extension=False):
        if keep and keep<1: raise ValueError('Checkpoint retention must be positive (or 0 for unlimited)')
        self.store=ArtifactStore(root)
        self.root,self.run_id,self.rank,self.world_size=str(root),run_id,rank,world_size
        self.device,self.contract,self.keep,self.cache=device,contract,keep,cache
        self.allow_horizon_extension=allow_horizon_extension
        if world_size>1 and not(self.store.cloud or shared):
            raise ValueError('Distributed checkpoints require gs:// or explicitly configured --checkpoint-shared filesystem; rank-local paths cannot be resumed across hosts')

    def verify_storage(self):
        require_equal(dict(root=self.root,run_id=self.run_id,world=self.world_size,contract=self.contract),'checkpoint-contract')
        name=f'probes/{self.run_id}/rank{self.rank}.json'
        self.store.put_json(name,dict(rank=self.rank,hostname=socket.gethostname(),nonce=uuid.uuid4().hex))
        names=gather_json(name,'checkpoint-storage-probes')
        visible=[self.store.exists(p) for p in names]
        status=gather_json(dict(rank=self.rank,visible=visible),'checkpoint-storage-visibility')
        if any(not all(item['visible']) for item in status):
            raise RuntimeError(f'Checkpoint destination is not actually shared across all hosts: {status}')

    def latest(self):
        steps=complete_steps(self.root)
        if not steps: raise FileNotFoundError(f'No complete checkpoints in {self.root}')
        return steps[-1]

    def load(self,step):
        return read_complete(self.root,step,self.rank,True,self.world_size,self.contract,self.cache,
                             allow_horizon_extension=self.allow_horizon_extension)

    def save(self,step,model,optimizer,metadata,runtime):
        # Already committed checkpoints are immutable; no overwrite on a resumed boundary.
        present=self.store.exists(f'complete_{step:06d}.json')
        require_equal(present,f'checkpoint-existing-{step}')
        if present:
            existing=self.store.get_json(f'complete_{step:06d}.json')
            if existing['run_id']!=self.run_id or existing['contract']!=self.contract:
                raise FileExistsError('Checkpoint destination contains another run/contract; choose a unique run-id/destination')
            return
        attempts=gather_json(uuid.uuid4().hex if self.rank==0 else '',f'checkpoint-attempt-{step}')
        prefix=f'attempts/{step:06d}-{attempts[0]}'
        optimizer_kind='replicated' if self.device.type=='xla' or self.world_size==1 else 'sharded'
        records={}; error=None
        with diag.span('checkpoint_write',checkpoint_step=step,destination=self.root):
            try:
                with tempfile.TemporaryDirectory(prefix='llwm-checkpoint-') as tmp:
                    def save(key,value,name):
                        path=Path(tmp)/name
                        with diag.span('checkpoint_to_cpu',artifact=key):
                            torch.save(cpu_tree(value),path)
                        record=file_record(path,f'{prefix}/{name}')
                        with diag.span('checkpoint_upload',artifact=key,bytes=record['size']):
                            self.store.put_file(record['name'],path,exclusive=True)
                        records[key]=record
                        path.unlink()  # bound temporary disk use to one artifact
                    if self.rank==0:
                        save('model',model,f'model_{step:06d}.pt')
                        path=Path(tmp)/'meta.json'
                        path.write_text(json.dumps(metadata,sort_keys=True))
                        record=file_record(path,f'{prefix}/meta.json')
                        self.store.put_file(record['name'],path,exclusive=True)
                        records['metadata']=record
                    if optimizer_kind=='sharded' or self.rank==0:
                        save('optimizer',optimizer,f'optim_rank{self.rank}.pt')
                    save('runtime',dict(runtime,rank=self.rank,step=step),f'runtime_rank{self.rank}.pt')
            except Exception as exc:
                import traceback
                error=dict(exception=repr(exc),traceback=traceback.format_exc())
            reports=gather_json(dict(rank=self.rank,error=error,files=records),f'checkpoint-receipts-{step}')
            if any(report['error'] for report in reports):
                # A failed transaction has no completion manifest and is never
                # resumable. Remove its successfully uploaded rank artifacts so
                # retries do not accumulate orphaned checkpoint files.
                cleanup_error=None
                if self.rank==0:
                    try:
                        for report in reports:
                            for record in report['files'].values():
                                self.store.delete(record['name'])
                    except Exception as exc:
                        cleanup_error=repr(exc)
                cleanup_errors=gather_json(cleanup_error,f'checkpoint-cleanup-{step}')
                if any(cleanup_errors):
                    diag.event('checkpoint_cleanup_error',checkpoint_step=step,errors=cleanup_errors)
                raise RuntimeError(f'Checkpoint transaction failed, no completion manifest written: {reports}')
            commit_error=None
            if self.rank==0:
                try:
                    first=reports[0]['files']
                    manifest=dict(format=FORMAT,step=step,run_id=self.run_id,world_size=self.world_size,
                                  optimizer_kind=optimizer_kind,contract=self.contract,model=first['model'],metadata=first['metadata'],
                                  runtime={str(p['rank']):p['files']['runtime'] for p in reports},
                                  optimizer={str(p['rank']):p['files']['optimizer'] for p in reports if 'optimizer' in p['files']})
                    self.store.put_json(f'complete_{step:06d}.json',manifest,exclusive=True)
                    self.store.put_json(f'meta_{step:06d}.json',metadata)  # small convenience index, not a completion signal
                    self.store.put_json('LATEST.json',dict(step=step,manifest=f'complete_{step:06d}.json'))
                except Exception as exc: commit_error=repr(exc)
            errors=gather_json(commit_error,f'checkpoint-committed-{step}')
            if any(errors): raise RuntimeError(f'Checkpoint commit error: {errors}')
            diag.event('checkpoint_complete',checkpoint_step=step,destination=self.root)
            if self.rank==0 and self.keep:
                try:
                    self.prune()
                except Exception as exc:
                    diag.event('checkpoint_retention_error',exception=repr(exc))

    def prune(self):
        steps=complete_steps(self.root)
        # Only this protocol/run's old completed checkpoints are eligible. Keep
        # the previous completed save until the newest save has its manifest;
        # legacy files and incomplete attempts are never deleted by retention.
        for step in steps[:-self.keep]:
            name=f'complete_{step:06d}.json'; manifest=self.store.get_json(name)
            if manifest['run_id']!=self.run_id: continue
            records=[manifest['model'],manifest['metadata'],*manifest['runtime'].values(),*manifest['optimizer'].values()]
            # Remove eligibility first, only after the new completion manifest exists.
            self.store.delete(name)
            for record in records: self.store.delete(record['name'])
            if self.store.exists(f'meta_{step:06d}.json'): self.store.delete(f'meta_{step:06d}.json')
        # This checkpoint directory is scoped to one run. Keep files referenced
        # by its retained completion manifests and remove failed/intermediate
        # attempt artifacts, so each successful save replaces the prior save.
        retained=set()
        for step in complete_steps(self.root)[-self.keep:]:
            manifest=self.store.get_json(f'complete_{step:06d}.json')
            if manifest.get('run_id')==self.run_id:
                retained.update(record['name'] for record in
                                [manifest['model'],manifest['metadata'],*manifest['runtime'].values(),*manifest['optimizer'].values()])
        for name in self.store.list('attempts/'):
            if name not in retained:
                self.store.delete(name)
