"""Failure/stop propagation across hosts through an explicitly shared run store."""
import os
from pathlib import Path
import socket
import time
import traceback
import uuid
from nanochat.storage import ArtifactStore


class RunControl:
    def __init__(self,root,run_id):
        if not run_id or '/' in run_id or run_id in ('.','..'):
            raise ValueError('run-id must be one nonempty path component')
        self.store=ArtifactStore(str(root).rstrip('/')+'/'+run_id)
        self.run_id=run_id

    def fail(self,reason,**fields):
        value=dict(run_id=self.run_id,hostname=socket.gethostname(),pid=os.getpid(),time=time.time(),reason=reason,**fields)
        name=f'failure-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}.json'
        self.store.put_json(name,value,exclusive=True)

    def problem(self):
        if self.store.exists('STOP.json'): return self.store.get_json('STOP.json')
        names=self.store.list('failure-')
        return self.store.get_json(names[0]) if names else None

    def stop(self):
        self.store.put_json('STOP.json',dict(run_id=self.run_id,reason='explicit user stop',time=time.time()))


def linux_start_ticks(pid):
    try:
        # comm can contain spaces/parentheses; starttime is field 22.
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return int(fields[19])
    except (OSError,IndexError,ValueError):
        return None
