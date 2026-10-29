"""Supervised PJRT launch with per-rank native logs and multi-host failure propagation."""
import argparse
import glob
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

from nanochat.diagnostics import enable_faults
from nanochat.run_control import RunControl, linux_start_ticks


def _worker(index,module,argv):
    directory=Path(os.environ['NANOCHAT_LAUNCH_LOGS'])
    enable_faults(directory,f'{socket.gethostname()}-pid{os.getpid()}-bootstrap',redirect=True)
    record=directory/f'worker-{os.getpid()}.json'
    def status(value,**fields):
        temporary=record.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(pid=os.getpid(),status=value,**fields)))
        os.replace(temporary,record)
    status('starting')
    try:
        import torch_xla.runtime as xr
        from nanochat import diagnostics as diag
        rank,local,world=xr.global_ordinal(),xr.local_ordinal(),xr.world_size()
        status('running',rank=rank,local_rank=local,world_size=world)
        logger=diag.configure(directory,rank,local,os.environ['NANOCHAT_RUN_ID'],
                              os.environ.get('NANOCHAT_DEBUG')=='1',float(os.environ.get('NANOCHAT_WATCHDOG_SECONDS','600')))
        expected=int(os.environ.get('NANOCHAT_EXPECTED_WORLD_SIZE','0'))
        if expected and world!=expected:
            raise RuntimeError(f'Expected {expected} distributed TPU devices, found {world}')
        sys.argv=[module,*argv]
        runpy.run_module(module,run_name='__main__')
        logger.event('worker_complete')
        status('complete',rank=rank)
    except BaseException as exc:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        try:
            RunControl(os.environ['NANOCHAT_CONTROL_DIR'],os.environ['NANOCHAT_LAUNCH_ID']).fail(
                repr(exc),traceback=traceback.format_exc(),worker_pid=os.getpid())
        except Exception:
            traceback.print_exc()
        status('failed',exception=repr(exc))
        raise


def terminate_group(process,grace=20):
    """Signal only the new session/group created by this supervisor."""
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError: return
    deadline=time.monotonic()+grace
    while time.monotonic()<deadline:
        try: os.killpg(process.pid,0)
        except ProcessLookupError: break
        time.sleep(0.2)
    try: os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError: pass
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired: pass


def supervise(command,environment,logs,control,poll=2,grace=20):
    """No XLA import in this process: child failures cannot strand its device handles."""
    logs=Path(logs); logs.mkdir(parents=True,exist_ok=True)
    if control.problem():
        raise RuntimeError('This launch-id has a STOP/failure record; use a fresh --launch-id for another attempt')
    # Fail before touching TPU devices if shared control is not writable.
    control.store.put_json(f'hosts/{socket.gethostname()}.json',dict(hostname=socket.gethostname(),pid=os.getpid(),time=time.time()))
    process=subprocess.Popen(command,env=environment,start_new_session=True)
    record=dict(run_id=environment['NANOCHAT_RUN_ID'],launch_id=environment['NANOCHAT_LAUNCH_ID'],
                hostname=socket.gethostname(),supervisor_pid=os.getpid(),worker_pgid=process.pid,
                worker_start_ticks=linux_start_ticks(process.pid))
    pidfile=logs/'launcher.json'; pidfile.write_text(json.dumps(record,indent=2))
    print(f'LLWM launcher: pid_file={pidfile}; rank logs={logs}',flush=True)
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'signal {signum}')
    old_handlers={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        while process.poll() is None:
            problem=control.problem()
            if problem:
                print('Run stop/failure detected: '+json.dumps(problem),file=sys.stderr,flush=True)
                terminate_group(process,grace)
                return 1
            for path in logs.glob('worker-*.json'):
                worker=json.loads(path.read_text())
                if worker['status'] in ('complete','failed'): continue
                try:
                    os.kill(worker['pid'],0)
                    stat=Path(f"/proc/{worker['pid']}/stat")
                    if stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0]=='Z':
                        raise ProcessLookupError()
                except ProcessLookupError:
                    control.fail('rank process disappeared without completion (native crash/OOM possible)',worker=worker)
                    terminate_group(process,grace)
                    return 1
            time.sleep(poll)
        if process.returncode:
            control.fail('PJRT launcher failed; original errors are in rank stdout/fault/JSON logs',exit_code=process.returncode)
        return process.returncode
    except BaseException as exc:
        control.fail('supervisor interrupted',exception=repr(exc))
        raise
    finally:
        terminate_group(process,grace)
        record['exit_code']=process.poll(); record['finished_at']=time.time()
        pidfile.write_text(json.dumps(record,indent=2))
        for sig,handler in old_handlers.items(): signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single-process',action='store_true')
    parser.add_argument('--run-id',default=os.environ.get('RUN_ID'))
    parser.add_argument('--launch-id',default=os.environ.get('ATTEMPT_ID'))
    parser.add_argument('--log-dir',default=os.environ.get('LOG_DIR','logs'))
    parser.add_argument('--control-dir',default=os.environ.get('CONTROL_DIR'))
    parser.add_argument('--expected-world-size',type=int,default=0)
    parser.add_argument('--expected-local-devices',type=int,default=0)
    parser.add_argument('--watchdog-seconds',type=float,default=600)
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--_child',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('module')
    parser.add_argument('args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    os.environ.setdefault('PJRT_DEVICE','TPU')
    os.environ.setdefault('XLA_NO_SPECIAL_SCALARS','1')
    os.environ['PYTHONUNBUFFERED']='1'
    os.environ['PYTHONFAULTHANDLER']='1'
    if os.environ['PJRT_DEVICE']!='TPU': parser.error('PJRT_DEVICE must be TPU')
    if os.environ.get('XLA_USE_SPMD')=='1': parser.error('This launcher uses replicated gradients; unset XLA_USE_SPMD')
    if args._child:
        if args.expected_local_devices:
            available=len(glob.glob('/dev/accel*'))
            if available!=args.expected_local_devices:
                parser.error(f'Expected {args.expected_local_devices} local accelerator devices; found {available}')
        import torch_xla
        torch_xla.launch(_worker,args=(args.module,args.args),debug_single_process=args.single_process)
        return
    if args.expected_world_size>1 and (not args.run_id or not args.control_dir):
        parser.error('Distributed reliability launches require identical --run-id and explicit shared --control-dir on every host')
    run_id=args.run_id or ('local-'+uuid.uuid4().hex[:12])
    launch_id=args.launch_id or run_id
    logs=Path(args.log_dir).resolve()/run_id/launch_id/socket.gethostname()
    control_dir=args.control_dir or str(Path(args.log_dir).resolve()/'control')
    environment=dict(os.environ,NANOCHAT_RUN_ID=run_id,NANOCHAT_LAUNCH_ID=launch_id,
                     NANOCHAT_LAUNCH_LOGS=str(logs),NANOCHAT_CONTROL_DIR=control_dir,
                     NANOCHAT_EXPECTED_WORLD_SIZE=str(args.expected_world_size),
                     NANOCHAT_WATCHDOG_SECONDS=str(args.watchdog_seconds),NANOCHAT_DEBUG='1' if args.debug else '0')
    argv=[sys.executable,'-u','-m','scripts.xla_launch','--_child']
    if args.single_process: argv+=['--single-process']
    if args.expected_local_devices: argv += [f'--expected-local-devices={args.expected_local_devices}']
    argv += [args.module,*args.args]
    raise SystemExit(supervise(argv,environment,logs,RunControl(control_dir,launch_id)))


if __name__=='__main__': main()
