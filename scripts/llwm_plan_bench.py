"""Benchmark the full checkpoint's MPC, optionally retaining training optimizer buffers."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from nanochat.common import compute_init, compute_cleanup, print0
from nanochat.checkpoint_manager import build_model, load_optimizer_state
from nanochat.distributed_utils import require_equal
from nanochat.planning import mpc_plan
from nanochat import diagnostics as diag


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',required=True)
    parser.add_argument('--step',type=int,default=500)
    parser.add_argument('--iterations',type=int,default=3)
    parser.add_argument('--horizon',type=int,default=20)
    parser.add_argument('--execute-steps',type=int,default=10)
    parser.add_argument('--prompt',default='What is the tallest mountain?')
    parser.add_argument('--goal',default='.')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    _,rank,_,world,device=compute_init('xla')
    require_equal('ready','plan-bench-ready')
    if rank==0:
        import torch_xla
        os.environ['NANOCHAT_CHECKPOINT_DIR']=args.checkpoint_dir
        model,tokenizer,_=build_model(args.checkpoint_dir,args.step,device,'eval')
        optimizer=model.setup_optimizer()
        state=load_optimizer_state('base',device,rank,step=args.step)
        optimizer.load_state_dict(state)
        del state
        torch_xla.sync(wait=True)
        diag.event('plan_bench_model_ready',metrics=diag.xla_metrics(device))
        print0('Full-size checkpoint and optimizer loaded; starting planning benchmark',flush=True)
        prompt=tokenizer.encode(args.prompt,prepend=tokenizer.get_bos_token_id())
        goal=tokenizer.encode(args.goal)
        started=time.monotonic()
        result=mpc_plan(model,torch.tensor([prompt],device=device),torch.tensor(goal,device=device),
                        horizon=args.horizon,execute_steps=args.execute_steps,
                        iterations=args.iterations,lr=.01,seed=123)
        report=dict(elapsed=time.monotonic()-started,world_size=world,
                    optimizer_buffers_loaded=True,round_history=result['round_history'],
                    history=result['history'],text=tokenizer.decode(result['tokens'][0].tolist()))
        Path(args.output).write_text(json.dumps(report,indent=2))
        diag.event('plan_bench_complete',**report,metrics=diag.xla_metrics(device))
        print0('Full-size planning benchmark complete: '+json.dumps(report),flush=True)
    require_equal('done','plan-bench-done')
    compute_cleanup()


if __name__=='__main__':
    main()
