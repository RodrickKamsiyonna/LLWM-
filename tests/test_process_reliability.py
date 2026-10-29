import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest

from nanochat.run_control import RunControl


def test_rank_failure_stops_other_host_group_without_touching_unrelated_process(tmp_path):
    # Two independent supervisors sharing a control namespace. TPU collective
    # failure handling is additionally covered by the real multi-host test.
    root=Path(__file__).resolve().parents[1]
    control=tmp_path/'control'
    unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(90)'])
    code='''import os,sys,time
from scripts.xla_launch import supervise
from nanochat.run_control import RunControl
mode=sys.argv[1]
command=[sys.executable,'-c','import sys,time; time.sleep(0.5); sys.exit(17)' if mode=='fail' else 'import time; time.sleep(90)']
env=dict(os.environ,NANOCHAT_RUN_ID='test',NANOCHAT_LAUNCH_ID='attempt')
raise SystemExit(supervise(command,env,sys.argv[3],RunControl(sys.argv[2],'attempt'),poll=0.05,grace=0.2))
'''
    processes=[]
    try:
        for mode in ('wait','fail'):
            processes.append(subprocess.Popen([sys.executable,'-c',code,mode,str(control),str(tmp_path/mode)],
                                              cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True))
        results=[process.communicate(timeout=20) for process in processes]
        assert all(process.returncode for process in processes)
        assert RunControl(control,'attempt').problem()
        assert unrelated.poll() is None
    finally:
        for process in processes:
            if process.poll() is None: process.kill(); process.wait()
        unrelated.terminate();unrelated.wait()


def test_native_failure_keeps_fault_information(tmp_path):
    root=Path(__file__).resolve().parents[1]
    code='from nanochat.diagnostics import enable_faults; import os; enable_faults('+repr(str(tmp_path))+',"rank5"); os.abort()'
    process=subprocess.run([sys.executable,'-c',code],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert process.returncode != 0
    assert 'Fatal Python error' in (tmp_path/'rank5.fault.log').read_text()


def test_watchdog_dumps_stacks_without_terminating_long_compilation(tmp_path):
    from nanochat.diagnostics import RankDiagnostics
    logger=RankDiagnostics(tmp_path,rank=0,watchdog_seconds=0.05)
    try:
        logger.event('xla_compiling')
        time.sleep(0.15)
        assert logger.fault and logger.fault.name
        text=Path(logger.fault.name).read_text()
        assert 'may still be progressing' in text
        assert not logger.closed.is_set()
    finally:logger.close()
