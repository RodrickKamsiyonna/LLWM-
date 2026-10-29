"""Real eight-device/two-host regression test (not a replacement training model).

Exercises accumulated gradients, exactly-once averaging, replicated updates,
changing scalar hyperparameters, complete shared checkpoints, and graph metrics.
The unchanged 1.3B model must separately pass the 300-step qualification profile.
"""
import argparse
import json
import os
import socket
import torch

from nanochat.common import compute_init,compute_cleanup,xla_microstep_boundary
from nanochat.distributed_utils import gather_json,accumulation_steps
from nanochat.optim import MuonAdamW
from nanochat.reliable_checkpoint import TrainingCheckpoints,capture_rng,restore_rng
from nanochat import diagnostics as diag


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--global-tokens',type=int,choices=[16384,131072],required=True)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--checkpoint-dir',required=True)
    parser.add_argument('--checkpoint-shared',action='store_true')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--save-every',type=int,default=25)
    parser.add_argument('--expected-hosts',type=int,default=2)
    parser.add_argument('--expected-world-size',type=int,default=8)
    parser.add_argument('--inject-failure-rank',type=int,default=-1)
    parser.add_argument('--inject-failure-step',type=int,default=3)
    parser.add_argument('--native-failure',action='store_true')
    parser.add_argument('--warmup-steps',type=int,default=30)
    parser.add_argument('--max-extra-compilations',type=int,default=16)
    parser.add_argument('--sync-mode',choices=['step','microstep','microstep-wait'],default='microstep')
    args=parser.parse_args()
    _,rank,local,world,device=compute_init('xla')
    hosts=gather_json(dict(host=socket.gethostname(),rank=rank),'test-host-topology')
    if world!=args.expected_world_size or len({h['host'] for h in hosts})!=args.expected_hosts:
        raise RuntimeError(f'This test requires the actual requested topology, got {hosts}')
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as metrics
    accumulation=accumulation_steps(args.global_tokens,1,1024,world)
    a=torch.nn.Parameter(torch.ones(8,device=device))
    matrix=torch.nn.Parameter(torch.arange(128,dtype=torch.float32).reshape(8,16).to(device)/100)
    optimizer=MuonAdamW([
        dict(kind='adamw',params=[a],lr=0.01,betas=(0.8,0.95),eps=1e-8,weight_decay=0.1),
        dict(kind='muon',params=[matrix],lr=0.01,momentum=0.95,beta2=0.9,ns_steps=5,weight_decay=0.02),
    ])
    reference=torch.nn.Parameter(torch.ones(8))
    reference_optimizer=torch.optim.AdamW([reference],lr=0.01,betas=(0.8,0.95),eps=1e-8,weight_decay=0.1,foreach=False)
    pattern=(torch.arange(128,dtype=torch.float32).reshape(8,16)+1)/128
    pattern_device=pattern.to(device)
    vector_pattern=torch.arange(1,9,dtype=torch.float32)/8
    vector_device=vector_pattern.to(device)
    average=(world+1)/2
    expected_momentum=torch.zeros(1,8,16)
    expected_first=torch.zeros(8)
    checkpoint=TrainingCheckpoints(args.checkpoint_dir,args.run_id,rank,world,device,
        dict(test='accumulation',world=world,global_tokens=args.global_tokens,steps=args.steps),
        shared=args.checkpoint_shared,cache=f'/tmp/llwm-test-cache/{args.run_id}')
    checkpoint.verify_storage()
    baseline_compiles=None
    reports=[]
    for step in range(1,args.steps+1):
        diag.set_context(step=step)
        if rank==args.inject_failure_rank and step==args.inject_failure_step:
            diag.event('injected_rank_failure',native=args.native_failure)
            if args.native_failure: os.abort()
            raise RuntimeError(f'injected rank failure {rank}')
        lr=0.01*(1+step*0.001)
        for group in optimizer.param_groups:group['lr']=lr
        reference_optimizer.param_groups[0]['lr']=lr
        for micro in range(accumulation):
            diag.set_context(microstep=micro)
            # Real stable-size input tensors on every microstep; each rank has
            # a different known gradient before the single replicated average.
            data=torch.full((1,1024),float(rank+1),dtype=torch.float32).to(device)
            loss=((a*vector_device).sum()+(matrix*pattern_device).sum())*data.mean()/accumulation
            loss.backward()
            xla_microstep_boundary(device,args.sync_mode)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch_xla.sync(wait=True)
        reference.grad=average*vector_pattern
        reference_optimizer.step();reference_optimizer.zero_grad(set_to_none=True)
        expected_first=0.8*expected_first+0.2*average*vector_pattern
        expected_momentum=0.95*expected_momentum+0.05*average*pattern.unsqueeze(0)
        if step==1 or step%25==0 or step==args.steps:
            torch.testing.assert_close(a.cpu(),reference.detach(),atol=2e-5,rtol=2e-5)
            actual_first=optimizer.state[a]['exp_avg'].cpu()
            torch.testing.assert_close(actual_first,expected_first,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(optimizer.state[matrix]['momentum_buffer'].cpu(),expected_momentum,atol=2e-5,rtol=2e-5)
            flat=torch.cat([a.detach().flatten(),matrix.detach().flatten()])
            replicas=xm.all_gather(flat.unsqueeze(0),dim=0)
            torch_xla.sync(wait=True)
            replicas=replicas.cpu()
            torch.testing.assert_close(replicas,replicas[:1].expand_as(replicas),atol=1e-6,rtol=1e-6)
            report=diag.xla_metrics(device)
            reports.append(dict(step=step,metrics=report))
            diag.event('distributed_test_metrics',metrics=report,global_tokens=args.global_tokens)
        if args.save_every>0 and step%args.save_every==0:
            rng=capture_rng(device)
            checkpoint.save(step,dict(a=a,matrix=matrix),optimizer.state_dict(),dict(step=step),
                            dict(rng=rng,loader=dict(rank=rank,offset=step*accumulation)))
            if step==args.save_every:
                # Mutate then restore both weights and moments. Continue training
                # against the same independent CPU AdamW reference afterward.
                weights,state,meta,runtime,_=checkpoint.load(step)
                with torch.no_grad():
                    a.add_(123);matrix.zero_()
                    a.copy_(weights['a']);matrix.copy_(weights['matrix'])
                optimizer.load_state_dict(state)
                restore_rng(runtime['rng'],device)
                if runtime['loader']['rank']!=rank or meta['step']!=step:
                    raise RuntimeError('Wrong cross-host rank checkpoint restored')
        compiled=metrics.metric_data('CompileTime')
        count=compiled[0] if compiled else 0
        if step==args.warmup_steps:baseline_compiles=count
        if baseline_compiles is not None and count-baseline_compiles>args.max_extra_compilations:
            raise RuntimeError(f'Unexpected continuing compilation: {baseline_compiles} -> {count} at {step}')
    summary=dict(passed=True,world_size=world,hosts=sorted({h['host'] for h in hosts}),rank=rank,
                 steps=args.steps,global_tokens=args.global_tokens,accumulation=accumulation,reports=reports)
    checkpoint.store.put_json(f'test-result-rank{rank}.json',summary)
    diag.event('distributed_test_passed',**{k:v for k,v in summary.items() if k!='reports'})
    print(json.dumps({k:v for k,v in summary.items() if k!='reports'}),flush=True)
    compute_cleanup()


if __name__=='__main__':main()
